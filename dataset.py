import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt_tab')


stop_words = set(stopwords.words('english'))



df = pd.read_csv("dataset.csv")

def clean_text(text):
    if isinstance(text, str):
        # Removes HTML tags, special characters, stopwords
        text = text.lower()
        text = re.sub(r'<.*?>', '', text) 
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Removes stopwords
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""


df['review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative':0})

df.to_csv("clean_dataset.csv", index=False)

