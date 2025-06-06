import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import joblib

# Wczytanie danych
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 0
real["label"] = 1

# Zbalansowanie danych: po 2500 z każdej klasy
fake_sample = fake.sample(n=2500, random_state=42)
real_sample = real.sample(n=2500, random_state=42)
data = pd.concat([fake_sample, real_sample], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Przygotowanie danych
X = data["text"]
y = data["label"]

# TF-IDF z ograniczeniem liczby cech
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)
X_vect = vectorizer.fit_transform(X)

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Trenowanie modeli
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Linear SVM": LinearSVC(max_iter=1000)
}

for name, model in models.items():
    print(f"\nTrenowanie modelu: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nWyniki dla modelu: {name}")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Zapis najlepszego modelu (np. Logistic Regression)
joblib.dump(models["Logistic Regression"], "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
