from starlette.background import BackgroundTask
import zipfile
from typing import Optional
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
    hotspots: str | None = Form(None),
    rfDiffusionDesigns: int = Form(...),
    designLoops: str = Form(""),
    rfDiffusionFinalStep: int = Form(48),
    rfDiffusionDeterministic: bool = Form(False),
    rfDiffusionDiffuserT: int = Form(50),
    proteinMPNNDesigns: int = Form(1),
    frameworkFile: UploadFile = File(...),
    targetFile: UploadFile = File(...),
) -> Dict[str, Any]:
    fw = save_upload(frameworkFile)
    tg = save_upload(targetFile)
    
    try:
        result = orchestrate_pipeline(
            job_name=jobName,
            mode=mode,
            hotspots=hotspots,
            rf_diffusion_designs=rfDiffusionDesigns,
            rf_diffusion_final_step=rfDiffusionFinalStep,
            rf_diffusion_deterministic=rfDiffusionDeterministic,
            rf_diffusion_diffuser_t=rfDiffusionDiffuserT,
            design_loops=designLoops,
            protein_mpnn_designs=proteinMPNNDesigns,
            framework_path_host=fw,
            target_path_host=tg,
        )
        job_id = result.get("jobId") or (result.get("job") or {}).get("jobId")
        if job_id:
            
            result.setdefault("links", {})
            result["links"]["download"] = {
                "jobZip":   f"/jobs/{job_id}/archive?scope=job",
                "outputZip":f"/jobs/{job_id}/archive?scope=output",
            }
        
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
def download_job_archive(job_id: str, scope: Optional[str] = "job"):
    job_dir = safe_job_dir(job_id)
    base_dir = job_dir if scope == "job" else (job_dir / "output")

    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"{scope} not found")

    try:
        next(base_dir.rglob("*"))
    except StopIteration:
        raise HTTPException(status_code=404, detail=f"no artifacts in {scope}")

    tmpdir = tempfile.mkdtemp()
    zip_path = Path(tmpdir) / f"{job_id}_{scope}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in base_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(job_dir))

    logger.info(f"[download] jobId={job_id} scope={scope} -> zip ready")

    bg = BackgroundTask(shutil.rmtree, tmpdir)
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{job_id}_{scope}.zip",
        headers={"Cache-Control": "no-store"},
        background=bg,
    )