"""Liveness, readiness and Prometheus metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.metrics import render_metrics

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe (checks the database)")
async def readyz(request: Request) -> Response:
    try:
        await request.app.state.db.ping()
    except Exception as exc:  # noqa: BLE001 - probe must not raise
        return Response(
            content=str({"status": "unavailable", "database": str(exc)[:200]}),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )
    return Response(content='{"status": "ok"}', media_type="application/json")


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")
