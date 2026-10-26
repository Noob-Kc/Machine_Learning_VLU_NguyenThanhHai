
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    # Tinh khoang cach Euclid tu mau moi den cac mau hoc
    khoang_cach = np.sqrt(
        np.sum((X_hoc - x_moi) ** 2, axis=1)
    )

    # Lay chi so cua k lang gieng gan nhat
    chi_so = np.argsort(khoang_cach)[:k]

    # Lay nhan cua k lang gieng
    nhan_lang_gieng = y_hoc[chi_so]

    # Bo phieu theo so luong nhan
    cac_nhan, so_phieu = np.unique(
        nhan_lang_gieng, return_counts=True
    )
    return cac_nhan[np.argmax(so_phieu)]


# Doc du lieu hai vang trang
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["nhan"].to_numpy()

# Chia tap hoc va tap kiem tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Du doan bang KNN tu viet
k = 13
y_du_doan = np.array([
    knn_du_doan(X_tr, y_tr, x, k)
    for x in X_te
])

acc_tu_viet = accuracy_score(y_te, y_du_doan)

# Du doan bang KNeighborsClassifier
mo_hinh = KNeighborsClassifier(n_neighbors=13)
mo_hinh.fit(X_tr, y_tr)
y_sklearn = mo_hinh.predict(X_te)

acc_sklearn = accuracy_score(y_te, y_sklearn)

print("BAI 1: TU VIET KNN")
print("So mau tap hoc:", len(X_tr))
print("So mau tap kiem tra:", len(X_te))
print(f"Accuracy KNN tu viet: {acc_tu_viet:.4f}")
print(f"Accuracy sklearn:     {acc_sklearn:.4f}")
print("Hai ket qua bang nhau:", np.isclose(
    acc_tu_viet, acc_sklearn
))
#Giải thích: Hàm tự viết tính khoảng cách từ từng mẫu kiểm tra đến toàn bộ mẫu học, lấy 13 mẫu gần nhất và bỏ phiếu đa số.
#Với bộ dữ liệu này, kết quả dự kiến là accuracy 0.9200, bằng sklearn theo kết quả mẫu của tài liệu. 
#Nếu không bằng, cần kiểm tra cách xử lý trường hợp hòa phiếu hoặc khoảng cách bằng nhau.