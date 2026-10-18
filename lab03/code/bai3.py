"""Bai 3: So sanh cac gia tri alpha cua MultinomialNB."""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print(f"{'alpha':>8} {'accuracy':>10} {'precision':>10} {'recall':>10}")
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    mo_hinh = MultinomialNB(alpha=alpha)
    mo_hinh.fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)
    print(f"{alpha:8.3f} {accuracy_score(y_test, y_pred):10.4f} "
          f"{precision_score(y_test, y_pred, zero_division=0):10.4f} "
          f"{recall_score(y_test, y_pred, zero_division=0):10.4f}")

print("\nNhan xet: alpha qua lon lam xac suat cac tu giua hai lop xich lai gan nhau.")
print("Vi vay, cac tu it dac trung hon va kha nang phan biet tin rac co the giam.")
