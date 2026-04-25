import logging
from datetime import datetime, timezone

import stripe
from sqlalchemy.orm import Session

from webapp.config import settings
from webapp.models import User, Subscription

log = logging.getLogger(__name__)

stripe.api_key = settings.STRIPE_SECRET_KEY


def create_checkout_session(user: User, success_url: str, cancel_url: str) -> str:
    customer_id = user.stripe_customer_id
    if not customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        customer_id = customer.id

    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": settings.STRIPE_PRICE_ID, "quantity": 1}],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"user_id": user.id},
    )
    return session.url


def create_portal_session(user: User, return_url: str) -> str:
    if not user.stripe_customer_id:
        raise ValueError("Stripe customer not found")
    session = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=return_url,
    )
    return session.url


def handle_webhook_event(event: dict, db: Session):
    event_type = event["type"]
    data = event["data"]["object"]
    now = datetime.now(timezone.utc).isoformat()

    if event_type == "checkout.session.completed":
        user_id = data.get("metadata", {}).get("user_id")
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")
        if not user_id or not subscription_id:
            return

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return

        user.stripe_customer_id = customer_id
        user.tier = "pro"
        user.updated_at = now

        sub = stripe.Subscription.retrieve(subscription_id)
        db_sub = Subscription(
            id=subscription_id,
            user_id=user_id,
            stripe_subscription_id=subscription_id,
            stripe_price_id=sub["items"]["data"][0]["price"]["id"],
            status="active",
            current_period_start=datetime.fromtimestamp(sub["current_period_start"], tz=timezone.utc).isoformat(),
            current_period_end=datetime.fromtimestamp(sub["current_period_end"], tz=timezone.utc).isoformat(),
            created_at=now,
            updated_at=now,
        )
        db.merge(db_sub)
        db.commit()

    elif event_type == "customer.subscription.deleted":
        sub_id = data.get("id")
        db_sub = db.query(Subscription).filter(Subscription.stripe_subscription_id == sub_id).first()
        if db_sub:
            db_sub.status = "canceled"
            db_sub.updated_at = now
            user = db.query(User).filter(User.id == db_sub.user_id).first()
            if user:
                user.tier = "free"
                user.updated_at = now
            db.commit()

    elif event_type == "invoice.payment_failed":
        sub_id = data.get("subscription")
        db_sub = db.query(Subscription).filter(Subscription.stripe_subscription_id == sub_id).first()
        if db_sub:
            db_sub.status = "past_due"
            db_sub.updated_at = now
            db.commit()
