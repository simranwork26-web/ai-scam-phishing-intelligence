from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.services.text_model import get_text_model
from backend.app.services.url_model import get_url_model
from src.features.manipulation_signals import detect_manipulation_signals
from src.features.extract_urls_from_text import URL_PATTERN, clean_url


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
    url_count: int
    url_max_probability: float
    url_mean_probability: float
    url_risk_level: str



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
    urls = [clean_url(url) for url in URL_PATTERN.findall(request.message)]
    url_model = get_url_model()
    url_probabilities = [url_model.predict(url) for url in urls]
    url_count = len(url_probabilities)
    url_max_probability = max(url_probabilities, default=0.0)
    url_mean_probability = sum(url_probabilities) / url_count if url_count else 0.0
    url_risk_level = (
        "high" if url_max_probability >= 0.90
        else "medium" if url_max_probability >= 0.50
        else "low"
    )


    return AnalyzeResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        model="distilbert_short_benign",
        device=str(model.device),
        signals=indicators["individual_signals"],
        combinations=indicators["combinations"],
        url_count=url_count,
        url_max_probability=url_max_probability,
        url_mean_probability=url_mean_probability,
        url_risk_level=url_risk_level,

    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "scam-phishing-intelligence",
    }
