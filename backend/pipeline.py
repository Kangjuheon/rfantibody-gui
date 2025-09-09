
import os, shutil, uuid, json, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import docker
from loguru import logger


JOBS_ROOT = Path(os.getenv("JOBS_ROOT", "/data/jobs"))


RF_WORKER_NAME = os.getenv("RF_WORKER_NAME", "rfantibody-worker")


SETUP_SENTINEL = "/home/.rf_setup_done"

def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_design_loops(user_str: str) -> str:
    s = user_str.strip()
    if not s:
        return ""
    items = [re.sub(r"\s+", "", it) for it in s.split(",") if it.strip()]
    valid = []
    for it in items:
        if re.match(r"^(H1|H2|H3|L1|L2|L3)(:(\d+(-\d+)?))?$", it):
            valid.append(it)
    return "[" + ",".join(valid) + "]" if valid else ""

def parse_hotspots(user_str: str) -> str:
    
    s = user_str.strip()
    if not s:
        return "[]"
    toks = [re.sub(r"\s+", "", t) for t in s.split(",") if t.strip()]
    expanded: List[str] = []
    for t in toks:
        m = re.match(r"^([A-Za-z])(\d+)-([A-Za-z])?(\d+)$", t)
        if m:
            ch1, a, ch2, b = m.group(1), int(m.group(2)), m.group(3) or m.group(1), int(m.group(4))
            if ch1.upper() != ch2.upper():
                
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            expanded += [f"{ch1.upper()}{idx}" for idx in range(lo, hi + 1)]
        else:
            m2 = re.match(r"^([A-Za-z])(\d+)$", t)
            if m2:
                expanded.append(f"{m2.group(1).upper()}{m2.group(2)}")
    
    return "[" + ",".join(expanded) + "]" if expanded else "[]"


def _stream_exec_logs(client: docker.DockerClient, container, exec_id: str, job_id: str, stage: str, logfile: Path, tail_limit: int = 10000) -> str:
    ensure_dirs(logfile.parent)
    tail_buf: List[bytes] = []
    max_bytes = tail_limit

    
    for chunk in client.api.exec_start(exec_id, stream=True, demux=True):
        
        out = (chunk[0] or b"") + (chunk[1] or b"")
        if not out:
            continue
        
        with logfile.open("ab") as f:
            f.write(out)
        
        try:
            text = out.decode("utf-8", "ignore").rstrip("\n")
        except Exception:
            text = str(out)
        if text:
            logger.info(f"[job={job_id}] [{stage}] {text}")

        tail_buf.append(out)
        while sum(len(x) for x in tail_buf) > max_bytes and len(tail_buf) > 1:
            tail_buf.pop(0)

    try:
        return b"".join(tail_buf).decode("utf-8", "ignore")
    except Exception:
        return ""

def exec_in_worker(cmd: str, job_dir: Path, job_id: str, stage: str) -> Tuple[int, str]:
    """
    이미 실행 중인 rfantibody-worker 컨테이너 내부에서 bash -lc '{cmd}' 를 실행하고
    로그를 {JOBS_ROOT}/{job_id}/logs/{stage}.log 로 기록한다.
    """
    client = docker.from_env()
    try:
        container = client.containers.get(RF_WORKER_NAME)
    except Exception as e:
        raise RuntimeError(f"worker container '{RF_WORKER_NAME}' not found or not running: {e}")

    logs_dir = job_dir / "logs"
    ensure_dirs(logs_dir)
    logfile = logs_dir / f"{stage}.log"

    full_cmd = ["/bin/bash", "-lc", cmd]
    logger.info(f"[job={job_id}] [{stage}] exec in {RF_WORKER_NAME}: {cmd}")

    
    exec_obj = client.api.exec_create(
        container.id,
        cmd=full_cmd,
        stdout=True,
        stderr=True,
        tty=False,
        environment={},  
        workdir="/home", 
    )
    exec_id = exec_obj.get("Id")
    
    tail_text = _stream_exec_logs(client, container, exec_id, job_id, stage, logfile)
    
    inspect = client.api.exec_inspect(exec_id)
    exit_code = int(inspect.get("ExitCode", 1))
    logger.info(f"[job={job_id}] [{stage}] exec exit code {exit_code}")
    return exit_code, tail_text


def orchestrate_pipeline(
    job_name: str,
    mode: str,
    hotspots: str,
    rf_diffusion_designs: int,
    protein_mpnn_designs: int,
    design_loops: str,
    framework_path_host: Path,
    target_path_host: Path,
) -> Dict[str, Any]:

    
    job_id = f"{job_name}_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_ROOT / job_id
    inp_dir = job_dir / "input"
    out_dir = job_dir / "output"
    ensure_dirs(inp_dir); ensure_dirs(out_dir)

    
    fw_host = inp_dir / framework_path_host.name
    tg_host = inp_dir / target_path_host.name
    shutil.move(str(framework_path_host), fw_host)
    shutil.move(str(target_path_host), tg_host)

    loops_arg = parse_design_loops(design_loops)
    hotspots_arg = parse_hotspots(hotspots)

    
    
    rfd_out_prefix = out_dir / "ab_des"
    ensure_dirs(rfd_out_prefix)  
    logger.info(f"[job={job_id}] RFdiffusion output prefix: {rfd_out_prefix}")

    rfd_cmd_parts = [
        "poetry run python /home/src/rfantibody/rfdiffusion/rfdiffusion_inference.py",
        "--config-name antibody",
        f"antibody.target_pdb={tg_host}",
        f"antibody.framework_pdb={fw_host}",
        "inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt",
        f"ppi.hotspot_res={hotspots_arg}",
        f"inference.num_designs={rf_diffusion_designs}",
        f"inference.output_prefix={rfd_out_prefix}",
    ]
    if loops_arg:
        rfd_cmd_parts.append(f"antibody.design_loops={loops_arg}")

    cmd1 = " ".join(map(str, rfd_cmd_parts))
    code1, log1_tail = exec_in_worker(cmd1, job_dir, job_id, stage="rfdiffusion")

    
    rfd_files = list(out_dir.glob("ab_des*"))
    if code1 != 0 or not rfd_files:
        return {
            "status": "error",
            "stage": "rfdiffusion",
            "jobId": job_id,
            "log_tail": (log1_tail or "")[-4000:],
            "note": f"expected outputs missing (found {len(rfd_files)})"
        }

    
    rfd_pack_cmd = "mkdir -p {od}/rfdiffusion && mv {od}/ab_des* {od}/rfdiffusion/ 2>/dev/null || true".format(od=out_dir)
    exec_in_worker(rfd_pack_cmd, job_dir, job_id, stage="collect_rfdiffusion")

    
    mpnn_in = out_dir / "rfdiffusion"   
    mpnn_out = out_dir / "proteinmpnn"
    ensure_dirs(mpnn_out)

    cmd2 = " ".join([
        "poetry run python /home/scripts/proteinmpnn_interface_design.py",
        f"-pdbdir {mpnn_in}",
        f"-outpdbdir {mpnn_out}",
        f"-numseq {protein_mpnn_designs}",
    ])
    code2, log2_tail = exec_in_worker(cmd2, job_dir, job_id, stage="proteinmpnn")
    if code2 != 0:
        return {
            "status": "error",
            "stage": "proteinmpnn",
            "jobId": job_id,
            "log_tail": (log2_tail or "")[-4000:],
        }

    
    rf2_inp = mpnn_out
    rf2_out = out_dir / "rf2"
    ensure_dirs(rf2_out)

    cmd3 = " ".join([
        "poetry run python /home/scripts/rf2_predict.py",
        f"input.pdb_dir={rf2_inp}",
        f"output.pdb_dir={rf2_out}",
    ])
    code3, log3_tail = exec_in_worker(cmd3, job_dir, job_id, stage="rf2")
    if code3 != 0:
        return {
            "status": "error",
            "stage": "rf2",
            "jobId": job_id,
            "log_tail": (log3_tail or "")[-4000:],
        }

    
    (out_dir / "heartbeat.txt").write_text("ok\n")

    has_any = any(out_dir.rglob("*"))
    logs_rel = f"/files/jobs/{job_id}/logs"
    artifacts = [f"/jobs/{job_id}/archive"]

    return {
        "status": "ok" if has_any else "empty",
        "jobId": job_id,
        "artifacts": artifacts,
        "logs": {
            "rfdiffusion_tail": (log1_tail or "")[-4000:],
            "proteinmpnn_tail": (log2_tail or "")[-4000:],
            "rf2_tail": (log3_tail or "")[-4000:],
            "logs_dir": logs_rel
        }
    }