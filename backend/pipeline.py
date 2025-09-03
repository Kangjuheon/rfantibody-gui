import os, shutil, uuid, json, re, time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import docker
from docker.types import DeviceRequest
from loguru import logger

JOBS_ROOT = Path(os.getenv("JOBS_ROOT", "/data/jobs"))
RF_IMAGE   = os.getenv("RFANTIBODY_IMAGE", "rfantibody:latest")
RF_WEIGHTS_HOST = os.getenv("RF_WEIGHTS_HOST", "./third_party/RFantibody/weights")

GPU_REQ = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

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
        m = re.match(r"^T(\d+)-T?(\d+)$", t, re.IGNORECASE)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            expanded += [f"T{idx}" for idx in range(lo, hi+1)]
        else:
            if re.match(r"^T\d+$", t, re.IGNORECASE):
                expanded.append("T" + re.findall(r"\d+", t)[0])
    return "[" + ",".join(expanded) + "]" if expanded else "[]"

def with_setup(cmd: str) -> str:
    # 최초 1회 컨테이너 내부 환경 구성
    return f"""
set -e
if [ ! -f {SETUP_SENTINEL} ]; then
  echo "[RFantibody setup] running include/setup.sh ..."
  bash /home/include/setup.sh && touch {SETUP_SENTINEL}
fi
{cmd}
"""

def _stream_container_logs(container, job_id: str, stage: str, logfile: Path, tail_limit: int = 10000) -> str:
    """
    컨테이너 로그를 실시간으로 Loguru 및 파일로 전달.
    반환값: tail 문자열(최대 tail_limit 바이트)
    """
    ensure_dirs(logfile.parent)
    tail_buf: List[bytes] = []
    max_bytes = tail_limit

    with logfile.open("ab") as f:
        try:
            for chunk in container.logs(stream=True, follow=True, stdout=True, stderr=True):
                if not chunk:
                    continue
                # 파일에 기록
                f.write(chunk)
                f.flush()
                # 백엔드 로그로도 출력(줄바꿈 정리)
                try:
                    text = chunk.decode("utf-8", "ignore").rstrip("\n")
                except Exception:
                    text = str(chunk)
                if text:
                    logger.info(f"[job={job_id}] [{stage}] {text}")

                # tail 버퍼 관리(메모리 과다 방지)
                tail_buf.append(chunk)
                # 누적 바이트 제한
                while sum(len(x) for x in tail_buf) > max_bytes and len(tail_buf) > 1:
                    tail_buf.pop(0)
        except Exception as e:
            logger.warning(f"[job={job_id}] [{stage}] log stream error: {e}")

    try:
        # 마지막으로 non-stream 모드 로그도 한 번 더(혹시 남은 버퍼)
        extra = container.logs(stdout=True, stderr=True)
        if extra:
            with logfile.open("ab") as f:
                f.write(extra)
            try:
                text = extra.decode("utf-8", "ignore").rstrip("\n")
            except Exception:
                text = str(extra)
            if text:
                logger.info(f"[job={job_id}] [{stage}] {text}")
            tail_buf.append(extra)
            while sum(len(x) for x in tail_buf) > max_bytes and len(tail_buf) > 1:
                tail_buf.pop(0)
    except Exception:
        pass

    # tail join
    try:
        return b"".join(tail_buf).decode("utf-8", "ignore")
    except Exception:
        return ""

def run_in_container(cmd: str, job_dir: Path, job_id: str, stage: str, mem: str = "10g") -> Tuple[int, str]:
    """
    컨테이너 실행 + 로그 스트리밍(Loguru/파일) + 종료 코드/테일 반환
    """
    client = docker.from_env()
    volumes = {
        str(job_dir): {"bind": "/home/job", "mode": "rw"},
        str(Path(RF_WEIGHTS_HOST).resolve()): {"bind": "/home/weights", "mode": "ro"},
    }
    logs_dir = job_dir / "logs"
    ensure_dirs(logs_dir)
    logfile = logs_dir / f"{stage}.log"

    logger.info(f"[job={job_id}] [{stage}] starting container")
    container = client.containers.run(
        image=RF_IMAGE,
        command=["/bin/bash","-lc", cmd],
        volumes=volumes,
        device_requests=GPU_REQ,
        mem_limit=mem,
        working_dir="/home",
        detach=True,
        tty=False,
        stdout=True,
        stderr=True,
    )

    # 실시간 로그 전달
    tail_text = _stream_container_logs(container, job_id, stage, logfile)

    # 종료 대기
    result = container.wait()
    exit_code = int(result.get("StatusCode", 1))
    logger.info(f"[job={job_id}] [{stage}] container exited with code {exit_code}")
    try:
        container.remove(force=True)
    except Exception:
        pass
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

    # 업로드 이동
    fw_host = inp_dir / framework_path_host.name
    tg_host = inp_dir / target_path_host.name
    shutil.move(str(framework_path_host), fw_host)
    shutil.move(str(target_path_host), tg_host)

    loops_arg = parse_design_loops(design_loops)
    hotspots_arg = parse_hotspots(hotspots)

    # ---------- (1) RFdiffusion ----------
    rfd_out_prefix = f"/home/job/output/ab_des"
    rfd_cmd = [
        "poetry run python /home/src/rfantibody/scripts/rfdiffusion_inference.py",
        "--config-name antibody",
        f"antibody.target_pdb=/home/job/input/{tg_host.name}",
        f"antibody.framework_pdb=/home/job/input/{fw_host.name}",
        "inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt",
        f"ppi.hotspot_res={hotspots_arg}",
    ]
    if loops_arg:
        rfd_cmd.append(f"antibody.design_loops={loops_arg}")
    rfd_cmd += [
        f"inference.num_designs={rf_diffusion_designs}",
        f"inference.output_prefix={rfd_out_prefix}"
    ]
    cmd1 = with_setup(" ".join(rfd_cmd))
    code1, log1_tail = run_in_container(cmd1, job_dir, job_id, stage="rfdiffusion")
    if code1 != 0:
        return {"status":"error","stage":"rfdiffusion","jobId": job_id, "log_tail": log1_tail}

    # 결과 모으기
    collect_cmd = "mkdir -p /home/job/output/rfdiffusion && mv /home/job/output/ab_des* /home/job/output/rfdiffusion/ 2>/dev/null || true"
    run_in_container(with_setup(collect_cmd), job_dir, job_id, stage="collect_rfdiffusion")

    # ---------- (2) ProteinMPNN ----------
    mpnn_in = "/home/job/output/rfdiffusion"
    mpnn_out = "/home/job/output/proteinmpnn"
    cmd2 = " ".join([
        "poetry run python /home/scripts/proteinmpnn_interface_design.py",
        f"-pdbdir {mpnn_in}",
        f"-outpdbdir {mpnn_out}",
        f"-numseq {protein_mpnn_designs}",
    ])
    cmd2 = with_setup(cmd2)
    code2, log2_tail = run_in_container(cmd2, job_dir, job_id, stage="proteinmpnn")
    if code2 != 0:
        return {"status":"error","stage":"proteinmpnn","jobId": job_id, "log_tail": log2_tail}

    # ---------- (3) RF2 ----------
    rf2_inp = mpnn_out
    rf2_out = "/home/job/output/rf2"
    cmd3 = " ".join([
        "poetry run python /home/scripts/rf2_predict.py",
        f"input.pdb_dir={rf2_inp}",
        f"output.pdb_dir={rf2_out}"
    ])
    cmd3 = with_setup(cmd3)
    code3, log3_tail = run_in_container(cmd3, job_dir, job_id, stage="rf2")
    if code3 != 0:
        return {"status":"error","stage":"rf2","jobId": job_id, "log_tail": log3_tail}

    # 디버그: 최소 파일 하나 보장(아카이브 동작 확인용)
    run_in_container("bash -lc 'echo ok > /home/job/output/heartbeat.txt'", job_dir, job_id, stage="post_heartbeat")

    # 산출물 존재 여부
    has_any = any(out_dir.rglob("*"))

    # 로그 파일 경로(다운로드/확인용)
    logs_rel = f"/files/jobs/{job_id}/logs"
    artifacts = [f"/jobs/{job_id}/archive"]  # 백엔드 라우트(/jobs/...) 기준

    return {
        "status": "ok" if has_any else "empty",
        "jobId": job_id,
        "artifacts": artifacts,
        "logs": {
            "rfdiffusion_tail": log1_tail[-4000:] if log1_tail else "",
            "proteinmpnn_tail": log2_tail[-4000:] if log2_tail else "",
            "rf2_tail": log3_tail[-4000:] if log3_tail else "",
            "logs_dir": logs_rel
        }
    }