from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

import stripe as stripe_lib

from webapp.config import settings
from webapp.dependencies import get_db, get_current_user
from webapp.models import User
from webapp.schemas import CheckoutResponse
from webapp.services.stripe_service import create_checkout_session, create_portal_session, handle_webhook_event

router = APIRouter(prefix="/api/billing", tags=["billing"])


@router.post("/checkout", response_model=CheckoutResponse)
def checkout(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.is_pro:
        raise HTTPException(400, "既にProプランです")
    base = str(request.base_url).rstrip("/")
    url = create_checkout_session(user, success_url=f"{base}/dashboard?upgraded=1", cancel_url=f"{base}/pricing")
    return CheckoutResponse(url=url)


@router.post("/portal", response_model=CheckoutResponse)
def portal(request: Request, user: User = Depends(get_current_user)):
    if not user.stripe_customer_id:
        raise HTTPException(400, "サブスクリプションがありません")
    base = str(request.base_url).rstrip("/")
    url = create_portal_session(user, return_url=f"{base}/dashboard")
    return CheckoutResponse(url=url)


@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe_lib.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe_lib.SignatureVerificationError):
        raise HTTPException(400, "Invalid signature")
    handle_webhook_event(event, db)
    return {"status": "ok"}
