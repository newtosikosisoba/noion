from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session
from webapp.database import SessionLocal
from webapp.models import User
from webapp.auth import decode_token


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(401, "ログインが必要です")
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(401, "無効なトークンです")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(401, "ユーザーが見つかりません")
    return user


def get_optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get("access_token")
    if not token:
        return None
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None
    return db.query(User).filter(User.id == payload["sub"]).first()
