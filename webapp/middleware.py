import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        key = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window
        hits = [t for t in self._hits[key] if t > window_start]
        if len(hits) >= self.max_requests:
            raise HTTPException(429, "リクエストが多すぎます。しばらくお待ちください。")
        hits.append(now)
        self._hits[key] = hits

        return await call_next(request)
