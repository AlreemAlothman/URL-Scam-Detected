from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from urllib.parse import urlparse

# التحميل
model = joblib.load('phishing_model.pkl')
cv = joblib.load('vectorizer.pkl')

app = FastAPI()

class URLInput(BaseModel):
    url: str

# القائمة البيضاء الموسعة
TRUSTED_DOMAINS = {
    'whatsapp.com', 'facebook.com', 'google.com', 'youtube.com', 'instagram.com',
    'twitter.com', 'linkedin.com', 'microsoft.com', 'apple.com', 'amazon.com',
    'netflix.com', 'github.com', 'paypal.com', 'wikipedia.org', 'imamu.edu.sa',
    'edu.sa', 'gov.sa', 'moe.gov.sa', 'kfupm.edu.sa', 'kaust.edu.sa'
}

def extract_base_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith('www.') else domain
    except:
        return url

def is_trusted_domain(url):
    domain = extract_base_domain(url)
    return domain in TRUSTED_DOMAINS

def model_predict(url):
    """التوقع باستخدام النموذج فقط"""
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.snowball import SnowballStemmer
    
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    stemmer = SnowballStemmer("english")
    
    tokens = tokenizer.tokenize(url)
    stemmed = [stemmer.stem(word) for word in tokens]
    processed = ' '.join(stemmed)
    
    vectorized = cv.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    confidence = 0.5
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(vectorized)[0].max()
    
    return {
        "prediction": "phishing" if prediction == 1 else "safe",
        "confidence": confidence
    }

@app.post("/predict")
def predict_phishing(data: URLInput):
    domain = extract_base_domain(data.url)
    
    # التحقق من القائمة البيضاء أولاً
    if is_trusted_domain(data.url):
        return {
            "url": data.url,
            "domain": domain,
            "prediction": "safe",
            "confidence": 0.99,
            "source": "trusted_domain_list",
            "reason": "الموقع موجود في قائمة المواقع العالمية الموثوقة"
        }
    
    # استخدام النموذج للمواقع غير المعروفة
    model_result = model_predict(data.url)
    
    return {
        "url": data.url,
        "domain": domain,
        "prediction": model_result["prediction"],
        "confidence": round(model_result["confidence"], 3),
        "source": "ml_model",
        "reason": "تم التحليل بواسطة نموذج machine learning"
    }

# endpoints للإدارة
@app.post("/add-trusted-domain")
def add_trusted_domain(domain: str):
    TRUSTED_DOMAINS.add(domain.lower())
    return {"message": f"تمت الإضافة: {domain}"}

@app.get("/trusted-domains")
def get_trusted_domains():
    return sorted(list(TRUSTED_DOMAINS))

