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

# ========= NLP و استخراج الدومين =========
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from urllib.parse import urlparse

# رجّعنا نفس التوكنيزر القديم (حروف فقط) عشان يطابق التدريب
tokenizer = RegexpTokenizer(r"[A-Za-z]+")
stemmer   = SnowballStemmer("english")

def preprocess_url(url: str) -> str:
    tokens = tokenizer.tokenize(url or "")
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)

# دومينات موثوقة فقط (لا تضيفي wixsite هنا)
TRUSTED_DOMAINS = {
    "google.com",
    "whatsapp.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "apple.com",
    "microsoft.com",
    "twitter.com",
    "x.com",
}

def get_domain(url: str) -> str:
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

        # 1️⃣ لو الدومين موثوق جدًا → نعتبره safe
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

        # 2️⃣ تجهيز النص للمودل (نفس ما درّبنا)
        processed = preprocess_url(url)
        X = vectorizer.transform([processed])

        # قرار المودل الأصلي (ما نغيّره)
        raw_y = int(model.predict(X)[0])   # 1 = phishing, 0 = safe

        # احتمال الفيشينج (معلومة إضافية)
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception:
                prob = None

        # 3️⃣ منطق التصنيف النهائي
        if raw_y == 1:
            # المودل متأكد أنه phishing → ما نحوله Safe أبداً
            label = "notsafe"
            safe_label = "not safe"

            # مستوى الخطورة من الاحتمال
            if prob is not None:
                if prob >= 0.9:
                    risk_level = "high"
                elif prob >= 0.7:
                    risk_level = "medium"
                else:
                    risk_level = "low"   # borderline بس ما زال notsafe
            else:
                risk_level = None

        else:
            # المودل قال safe
            label = "safe"
            safe_label = "safe"
            risk_level = "low"

        return {
            "url": url,
            "label": label,             # safe / notsafe
            "safe_label": safe_label,   # نص للواجهة
            "score": prob,              # احتمال الفيشينج
            "raw_model_label": raw_y,   # 0 أو 1 من المودل
            "risk_level": risk_level,   # high / medium / low
            "reason": "model_score",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
