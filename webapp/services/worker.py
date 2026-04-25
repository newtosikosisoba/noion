import logging
from concurrent.futures import ThreadPoolExecutor
from webapp.config import settings
from webapp.database import SessionLocal
from webapp.services.transcribe import run_transcription

log = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_progress_store: dict[str, dict] = {}


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
    return _executor


def get_progress(job_id: str) -> dict | None:
    return _progress_store.get(job_id)


def submit_job(job_id: str, input_path: str, output_dir: str, tier: str):
    _progress_store[job_id] = {"progress": 0, "message": "キューに追加されました"}

    def on_progress(msg, pct=None):
        _progress_store[job_id] = {"progress": pct or 0, "message": msg}

    def task():
        db = SessionLocal()
        try:
            run_transcription(job_id, input_path, output_dir, tier, on_progress, db)
        except Exception:
            log.exception("Worker task failed for job %s", job_id)
        finally:
            db.close()
            _progress_store.pop(job_id, None)

    get_executor().submit(task)


def shutdown():
    global _executor
    if _executor:
        _executor.shutdown(wait=False)
        _executor = None
