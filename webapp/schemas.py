from pydantic import BaseModel, EmailStr
from typing import Optional


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: Optional[str]
    tier: str
    created_at: str


class JobCreateResponse(BaseModel):
    job_id: str


class JobOutputs(BaseModel):
    midi_full: Optional[str] = None
    midi_piano: Optional[str] = None
    midi_bass: Optional[str] = None
    midi_drums: Optional[str] = None
    wav: Optional[str] = None


class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: int
    progress_message: Optional[str]
    original_filename: str
    tier_at_creation: str
    outputs: Optional[JobOutputs] = None
    created_at: str
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int


class CheckoutResponse(BaseModel):
    url: str
