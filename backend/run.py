"""FastAPI entrypoint. Run: uvicorn run:app --reload --app-dir backend (from repo root)."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from analysis import analyze_batch

app = FastAPI(title="Review Intelligence Pipeline", version="0.1.0")


class TextsIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze/sentiment")
def sentiment(body: TextsIn) -> dict:
    return {"results": analyze_batch(body.texts)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
