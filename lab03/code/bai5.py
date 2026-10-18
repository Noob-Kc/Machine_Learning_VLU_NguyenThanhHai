"""Bai 5: So sanh MultinomialNB va GaussianNB tren du lieu sinh vien."""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

for ten, mo_hinh in [
    ("MultinomialNB", MultinomialNB()),
    ("GaussianNB", GaussianNB())
]:
    mo_hinh.fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)
    print(f"{ten} accuracy = {accuracy_score(y_test, y_pred):.4f}")

print("\nNhan xet: MultinomialNB van chay vi du lieu khong am, nhung khong phu hop ve nguyen tac.")
print("Hai cot la so lieu lien tuc; GaussianNB phu hop hon vi mo ta moi lop bang phan phoi chuan.")
