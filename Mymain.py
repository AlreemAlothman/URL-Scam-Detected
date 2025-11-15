from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib

# ========= إعدادات التطبيق =========
app = FastAPI(title="URL Scam Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= تحميل الموديلات =========
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "phishing_model.pkl"
VECT_PATH  = BASE / "vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model/vectorizer: {e}")

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from urllib.parse import urlparse   

tokenizer = RegexpTokenizer(r"[A-Za-z0-9]+")
stemmer   = SnowballStemmer("english")

def preprocess_url(url: str) -> str:
    tokens = tokenizer.tokenize(url or "")
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)

# ========= دومينات موثوقة (تقدرين تزودينها لاحقًا) =========
TRUSTED_DOMAINS = {
    "google.com",
    "whatsapp.com",
    "facebook.com",
    "instagram.com",
    "apple.com",
    "microsoft.com",
    "twitter.com",
    "x.com",
}

def get_domain(url: str) -> str:
    """
    ترجّع الدومين الرئيسي مثل whatsapp.com أو google.com
    """
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host

# ========= نماذج الإدخال =========
class URLInput(BaseModel):
    url: str

# ========= مسارات مساعدة =========
@app.get("/")
def root():
    return {"message": "URL Scam Detector is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ========= التنبؤ (GET) =========
@app.get("/predict")
def predict_get(url: str = Query(..., description="URL to classify")):
    return _predict(url)

# ========= التنبؤ (POST) =========
@app.post("/predict")
def predict_post(data: URLInput):
    return _predict(data.url)

# ========= الدالة المشتركة =========
def _predict(url: str):
    try:
        url = url.strip()
        domain = get_domain(url)

        if domain in TRUSTED_DOMAINS:
            return {
                "url": url,
                "label": "safe",         
                "safe_label": "safe",    
                "score": 0.01,           
                "raw_model_label": 0,    
                "risk_level": "low",
                "reason": "trusted_domain",
            }

        processed = preprocess_url(url)
        X = vectorizer.transform([processed])

        raw_y = int(model.predict(X)[0])

        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[0, 1]) 
            except Exception:
                prob = None

        if prob is not None:
            HIGH = 0.90   
            MID  = 0.60   

            if prob >= HIGH:
                label = "notsafe"
                safe_label = "not safe"
                risk_level = "high"
            elif prob >= MID:
                label = "suspicious"
                safe_label = "suspicious"
                risk_level = "medium"
            else:
                label = "safe"
                safe_label = "safe"
                risk_level = "low"
        else:
            label = "notsafe" if raw_y == 1 else "safe"
            safe_label = "not safe" if raw_y == 1 else "safe"
            risk_level = None

        return {
            "url": url,
            "label": label,             # safe / suspicious / notsafe
            "safe_label": safe_label,   
            "score": prob,              
            "raw_model_label": raw_y,  
            "risk_level": risk_level,   
            "reason": "model_score",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
