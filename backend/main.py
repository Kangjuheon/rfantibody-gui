from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Dict, Any
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
import shutil, os, tempfile

JOBS_ROOT = Path(os.getenv("JOBS_ROOT", "/data/jobs"))

app = FastAPI()


app.mount("/files/jobs", StaticFiles(directory="/data/jobs"), name="jobfiles")

def save_upload(u: UploadFile) -> Path:
    suffix = "_" + u.filename.replace("/", "_")
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as out:
        shutil.copyfileobj(u.file, out)
    return Path(tmp.name)

from pipeline import orchestrate_pipeline  

@app.post("/rfantibody_pipeline")
async def rfantibody_pipeline(
    jobName: str = Form(...),
    mode: str = Form(...),
    hotspots: str = Form(...),
    rfDiffusionDesigns: int = Form(...),
    proteinMPNNDesigns: int = Form(...),
    designLoops: str = Form(""),
    frameworkFile: UploadFile = File(...),
    targetFile: UploadFile = File(...),
) -> Dict[str, Any]:
    fw = save_upload(frameworkFile)
    tg = save_upload(targetFile)
    logger.info(f"[submit] jobName={jobName} mode={mode} RF={rfDiffusionDesigns} MPNN={proteinMPNNDesigns}")
    try:
        result = orchestrate_pipeline(
            job_name=jobName,
            mode=mode,
            hotspots=hotspots,
            rf_diffusion_designs=rfDiffusionDesigns,
            protein_mpnn_designs=proteinMPNNDesigns,
            design_loops=designLoops,
            framework_path_host=fw,
            target_path_host=tg,
        )
        
        if result.get("status") == "error":
            stage = result.get("stage")
            tail  = result.get("log_tail","")
            logger.error(f"[pipeline-error] stage={stage}\n{tail}")
        else:
            logger.info(f"[pipeline-done] jobId={result.get('jobId')} status={result.get('status')}")
        return result
    except Exception as e:
        logger.exception(f"Exception during pipeline execution: {e}")
        raise
    finally:
        
        for p in (fw, tg):
            try:
                if p.exists() and p.parent == Path("/tmp"):
                    p.unlink()
            except Exception:
                pass

def safe_job_dir(job_id: str) -> Path:
    p = (JOBS_ROOT / job_id).resolve()
    if not str(p).startswith(str(JOBS_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="invalid job id")
    return p

@app.get("/jobs/{job_id}/archive")
def download_job_archive(job_id: str):
    job_dir = safe_job_dir(job_id)
    out_dir = job_dir / "output"
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="output not found")
    
    try:
        next(out_dir.rglob("*"))
    except StopIteration:
        raise HTTPException(status_code=404, detail="no artifacts")

    tmpdir = tempfile.mkdtemp()
    zip_path = Path(tmpdir) / f"{job_id}_output.zip"
    
    shutil.make_archive(zip_path.with_suffix(""), "zip", root_dir=out_dir)
    logger.info(f"[download] jobId={job_id} -> zip ready")
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{job_id}_output.zip",
        headers={"Cache-Control": "no-store"}
    )