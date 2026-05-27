from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from webapp.config import settings
from pathlib import Path

Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
db_path = settings.DATABASE_URL.replace("sqlite:///", "")
Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db():
    from webapp import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate(engine)


def _migrate(eng):
    with eng.connect() as conn:
        cols = {row[1] for row in conn.execute(__import__("sqlalchemy").text("PRAGMA table_info(jobs)"))}
        if "mp3_path" not in cols:
            conn.execute(__import__("sqlalchemy").text("ALTER TABLE jobs ADD COLUMN mp3_path TEXT"))
            conn.commit()
