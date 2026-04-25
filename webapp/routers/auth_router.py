from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import uuid

from webapp.dependencies import get_db, get_current_user
from webapp.models import User
from webapp.schemas import RegisterRequest, LoginRequest, UserResponse
from webapp.auth import hash_password, verify_password, create_access_token, create_refresh_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_auth_cookies(response: Response, user_id: str):
    access = create_access_token(user_id)
    refresh = create_refresh_token(user_id)
    response.set_cookie("access_token", access, httponly=True, samesite="lax", max_age=30 * 60)
    response.set_cookie("refresh_token", refresh, httponly=True, samesite="lax", max_age=7 * 24 * 3600)


@router.post("/register", response_model=UserResponse)
def register(req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(409, "このメールアドレスは既に登録されています")
    if len(req.password) < 8:
        raise HTTPException(400, "パスワードは8文字以上必要です")

    now = datetime.now(timezone.utc).isoformat()
    user = User(
        id=str(uuid.uuid4()),
        email=req.email,
        password_hash=hash_password(req.password),
        display_name=req.display_name,
        tier="free",
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    _set_auth_cookies(response, user.id)
    return UserResponse(id=user.id, email=user.email, display_name=user.display_name, tier=user.tier, created_at=user.created_at)


@router.post("/login", response_model=UserResponse)
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "メールアドレスまたはパスワードが正しくありません")

    _set_auth_cookies(response, user.id)
    return UserResponse(id=user.id, email=user.email, display_name=user.display_name, tier=user.tier, created_at=user.created_at)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "ログアウトしました"}


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return UserResponse(id=user.id, email=user.email, display_name=user.display_name, tier=user.tier, created_at=user.created_at)
