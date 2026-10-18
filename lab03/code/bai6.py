"""Bai 6: Danh gia bo loc khi lop rac hiem."""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
thuong = df[df["la_rac"] == 0]          # giu toan bo 180 tin thuong
rac = df[df["la_rac"] == 1].head(20)    # chi lay 20 tin rac dau tien
df_moi = pd.concat([thuong, rac], ignore_index=True)

print("So tin thuong:", int((df_moi["la_rac"] == 0).sum()))
print("So tin rac:", int((df_moi["la_rac"] == 1).sum()))

vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df_moi["noi_dung"])
y = df_moi["la_rac"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
print("Recall lop rac:", f"{recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print("Ma tran nham lan [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print("\nNhan xet: accuracy co the cao do tin thuong chiem da so, nhung recall cho biet")
print("bo loc bat duoc bao nhieu tin rac thuc su. Can xem ca recall, khong chi accuracy.")
