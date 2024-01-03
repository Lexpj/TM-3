import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1 − Collecting the Dataset
df = pd.read_csv('/content/sample_data/IMDB_Dataset.csv')

# Step 2− Preprocessing the Data
corpus = []
stemmer = PorterStemmer()
for i in range(0, len(df)):
   review = re.sub('[^a-zA-Z]', ' ', df['review'][i])
   review = review.lower()
   review = review.split()
   review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
   review = ' '.join(review)
   corpus.append(review)
   
# Step 3− Creating the TF-IDF Matrix
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Step 4− Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5− Training the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6− Evaluating the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1 score: {f1:}")
