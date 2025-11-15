from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

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

def preprocess_url(url):
    tokens = tokenizer.tokenize(url)
    stemmed = [stemmer.stem(word) for word in tokens]
    sent = ' '.join(stemmed)
    return sent

@app.post("/predict")
def predict_phishing(data: URLInput):
    processed = preprocess_url(data.url)
    vectorized = cv.transform([processed])
    prediction = model.predict(vectorized)
    return {"url": data.url, "prediction": "phishing" if prediction[0] == 1 else "safe"}
