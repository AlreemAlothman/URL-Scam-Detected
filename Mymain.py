from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from urllib.parse import urlparse
import re

# Load saved model and vectorizer
model = joblib.load('phishing_model.pkl')
cv = joblib.load('vectorizer.pkl')

# Setup tokenizer and stemmer
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

app = FastAPI()

# Input model for request
class URLInput(BaseModel):
    url: str

def extract_domain_features(url):
    """استخراج ميزات إضافية من الرابط لتحسين الدقة"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        features = {
            'domain_length': len(domain),
            'num_dots': domain.count('.'),
            'num_hyphens': domain.count('-'),
            'has_https': 1 if parsed.scheme == 'https' else 0,
            'path_length': len(parsed.path),
            'query_length': len(parsed.query),
            'has_at_symbol': 1 if '@' in url else 0,
            'num_special_chars': len(re.findall(r'[~!@#$%^&*()_+={}\[\]:;<>?/\\|]', url))
        }
        return features
    except:
        return {}

def preprocess_url(url):
    """معالجة محسنة للرابط مع الحفاظ على الهيكل"""
    try:
        # استخراج الدومين فقط للمعالجة الأساسية
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # تنظيف الدومين
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # استخدام الدومين المعالج مع بعض ميزات المسار
        domain_tokens = tokenizer.tokenize(domain)
        domain_stemmed = [stemmer.stem(word) for word in domain_tokens]
        
        # إضافة بعض ميزات المسار إذا كانت طويلة (مشبوهة)
        path = parsed.path.lower()
        if len(path) > 30:  # إذا كان المسار طويلاً
            path_tokens = tokenizer.tokenize(path[:50])  # أول 50 حرف فقط
            path_stemmed = [stemmer.stem(word) for word in path_tokens]
            domain_stemmed.extend(path_stemmed)
        
        sent = ' '.join(domain_stemmed)
        return sent
    except:
        # إذا فشل التحليل، استخدم المعالجة الأصلية
        tokens = tokenizer.tokenize(url)
        stemmed = [stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed)

@app.post("/predict")
def predict_phishing(data: URLInput):
    """التوقع مع معالجة محسنة للدومين"""
    
    # استخراج الميزات الإضافية
    features = extract_domain_features(data.url)
    
    # المعالجة المسبقة
    processed = preprocess_url(data.url)
    vectorized = cv.transform([processed])
    
    # التوقع
    prediction = model.predict(vectorized)
    
    # حساب درجة الثقة إذا كان النموذج يدعم
    confidence_score = 0.5
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized)[0]
            confidence_score = max(proba)
    except:
        pass
    
    # تحديد مستوى الثقة
    if confidence_score > 0.8:
        confidence_level = "high"
    elif confidence_score > 0.6:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    result = "phishing" if prediction[0] == 1 else "safe"
    
    return {
        "url": data.url,
        "prediction": result,
        "confidence": confidence_level,
        "confidence_score": round(confidence_score, 3),
        "domain": urlparse(data.url).netloc if '://' in data.url else data.url
    }

# نقطة نهاية لمعاينة المعالجة
@app.post("/debug-preprocess")
def debug_preprocess(data: URLInput):
    """لتصحيح عملية المعالجة المسبقة"""
    processed = preprocess_url(data.url)
    features = extract_domain_features(data.url)
    
    return {
        "original_url": data.url,
        "processed_text": processed,
        "domain_features": features,
        "domain": urlparse(data.url).netloc if '://' in data.url else data.url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
