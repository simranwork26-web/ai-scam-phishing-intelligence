from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.services.text_model import get_text_model
from src.features.manipulation_signals import detect_manipulation_signals


app = FastAPI(
    title="AI Scam & Phishing Intelligence API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    message: str


class AnalyzeResponse(BaseModel):
    risk_score: float
    risk_level: str
    model: str
    device: str
    signals: list[str]
    combinations: list[str]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    model = get_text_model()
    risk_score = model.predict(request.message)

    risk_level = (
        "high"
        if risk_score >= 0.90
        else "medium"
        if risk_score >= 0.50
        else "low"
    )

    indicators = detect_manipulation_signals(request.message)

    return AnalyzeResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        model="distilbert_short_benign",
        device=str(model.device),
        signals=indicators["individual_signals"],
        combinations=indicators["combinations"],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "scam-phishing-intelligence",
    }
