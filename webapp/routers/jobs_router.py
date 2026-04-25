import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from webapp.config import settings
from webapp.dependencies import get_db, get_current_user, get_optional_user
from webapp.models import User, Job
from webapp.schemas import JobCreateResponse, JobStatusResponse, JobOutputs, JobListResponse
from webapp.services.worker import submit_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=JobCreateResponse)
async def create_job(
    file: UploadFile = File(...),
    user: User | None = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(400, "ファイル名がありません")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"未対応ファイル形式: {suffix}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"ファイルサイズが上限({settings.MAX_UPLOAD_SIZE_MB}MB)を超えています")

    job_id = str(uuid.uuid4())
    tier = user.tier if user else "free"
    now = datetime.now(timezone.utc).isoformat()

    job = Job(
        id=job_id,
        user_id=user.id if user else None,
        status="pending",
        progress=0,
        original_filename=file.filename,
        file_size=len(content),
        tier_at_creation=tier,
        created_at=now,
    )
    db.add(job)
    db.commit()

    output_dir = Path(settings.UPLOAD_DIR) / "jobs" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = str(output_dir / f"input{suffix}")
    with open(input_path, "wb") as f:
        f.write(content)

    submit_job(job_id, input_path, str(output_dir), tier)
    return JobCreateResponse(job_id=job_id)


def _build_outputs(job: Job) -> JobOutputs | None:
    if job.status != "completed":
        return None
    return JobOutputs(
        midi_full=f"/api/jobs/{job.id}/download/midi_full" if job.midi_full_path else None,
        midi_piano=f"/api/jobs/{job.id}/download/midi_piano" if job.midi_piano_path else None,
        midi_bass=f"/api/jobs/{job.id}/download/midi_bass" if job.midi_bass_path else None,
        midi_drums=f"/api/jobs/{job.id}/download/midi_drums" if job.midi_drums_path else None,
        wav=f"/api/jobs/{job.id}/download/wav" if job.wav_path else None,
    )


def _job_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress or 0,
        progress_message=job.progress_message,
        original_filename=job.original_filename,
        tier_at_creation=job.tier_at_creation,
        outputs=_build_outputs(job),
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("", response_model=JobListResponse)
def list_jobs(
    skip: int = 0,
    limit: int = 20,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Job).filter(Job.user_id == user.id).order_by(Job.created_at.desc())
    total = query.count()
    jobs = query.offset(skip).limit(limit).all()
    return JobListResponse(jobs=[_job_response(j) for j in jobs], total=total)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "ジョブが見つかりません")
    return _job_response(job)


_DOWNLOAD_MAP = {
    "midi_full": ("midi_full_path", "audio/midi", ".mid"),
    "midi_piano": ("midi_piano_path", "audio/midi", ".mid"),
    "midi_bass": ("midi_bass_path", "audio/midi", ".mid"),
    "midi_drums": ("midi_drums_path", "audio/midi", ".mid"),
    "wav": ("wav_path", "audio/wav", ".wav"),
}


@router.get("/{job_id}/download/{file_type}")
def download(job_id: str, file_type: str, db: Session = Depends(get_db)):
    if file_type not in _DOWNLOAD_MAP:
        raise HTTPException(400, f"無効なファイルタイプ: {file_type}")

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "ジョブが見つかりません")
    if job.status != "completed":
        raise HTTPException(400, "ジョブがまだ完了していません")

    attr, media_type, ext = _DOWNLOAD_MAP[file_type]
    file_path = getattr(job, attr)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, "ファイルが見つかりません")

    stem = Path(job.original_filename).stem
    download_name = f"{stem}_{file_type}{ext}"
    return FileResponse(path=file_path, media_type=media_type, filename=download_name)
