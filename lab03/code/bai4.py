"""Bai 4: Tu cai dat Multinomial Naive Bayes bang tong loga."""
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"]).toarray()
y = df["la_rac"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

alpha = 1.0
cac_lop = np.unique(y_train)
so_tu = X_train.shape[1]
log_prior = {}
log_xac_suat_tu = {}

for lop in cac_lop:
    X_lop = X_train[y_train == lop]
    log_prior[lop] = np.log(len(X_lop) / len(X_train))
    dem_tu = X_lop.sum(axis=0)
    tong_tu = dem_tu.sum()
    # Lam tron Laplace: cong alpha vao moi tu
    log_xac_suat_tu[lop] = np.log((dem_tu + alpha) / (tong_tu + alpha * so_tu))

def du_doan(X_can_doan):
    ket_qua = []
    for dong in X_can_doan:
        diem = {
            lop: log_prior[lop] + np.dot(dong, log_xac_suat_tu[lop])
            for lop in cac_lop
        }
        ket_qua.append(max(diem, key=diem.get))
    return np.array(ket_qua)

y_pred_tu_viet = du_doan(X_test)
print("Bo loc tu viet (tong loga) accuracy:", f"{accuracy_score(y_test, y_pred_tu_viet):.4f}")

# So sanh voi thu vien sklearn tren cung tap train/test
thu_vien = MultinomialNB(alpha=alpha)
thu_vien.fit(X_train, y_train)
y_pred_sklearn = thu_vien.predict(X_test)
print("MultinomialNB sklearn accuracy       :", f"{accuracy_score(y_test, y_pred_sklearn):.4f}")
print("So mau du doan khac nhau              :", int(np.sum(y_pred_tu_viet != y_pred_sklearn)))
print("Nhan xet: hai accuracy can bang nhau; neu so mau khac nhau bang 0 thi hai bo loc du doan giong nhau.")
