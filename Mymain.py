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
    # لو صار خطأ بالتحميل خلّيه واضح في اللوق
    raise RuntimeError(f"Failed to load model/vectorizer: {e}")

# ========= تجهيز الـ NLP (بدون تنزيل كوربسات) =========
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

tokenizer = RegexpTokenizer(r"[A-Za-z]+")
stemmer   = SnowballStemmer("english")

def preprocess_url(url: str) -> str:
    tokens = tokenizer.tokenize(url or "")
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)

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
        processed = preprocess_url(url)
        X = vectorizer.transform([processed])
        y = model.predict(X)[0]

        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception:
                prob = None

        return {
            "url": url,
            "label": int(y),                       # 1=phishing, 0=safe
            "prediction": "phishing" if y == 1 else "safe",
            "probability": prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

