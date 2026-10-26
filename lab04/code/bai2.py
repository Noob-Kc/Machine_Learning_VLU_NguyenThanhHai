
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Chon k bang kiem dinh cheo 5 phan
diem = {}

print("DIEM KIEM DINH CHEO - MANHATTAN")
for k in range(1, 52, 2):
    mo_hinh = KNeighborsClassifier(
        n_neighbors=k,
        metric="manhattan"
    )
    diem[k] = cross_val_score(
        mo_hinh, X_tr, y_tr, cv=5
    ).mean()

    print(f"k = {k:2d}: {diem[k]:.4f}")

k_tot = max(diem, key=diem.get)

# Danh gia tren tap kiem tra sau khi chon k
mo_hinh_tot = KNeighborsClassifier(
    n_neighbors=k_tot,
    metric="manhattan"
)
mo_hinh_tot.fit(X_tr, y_tr)

y_du_doan = mo_hinh_tot.predict(X_te)
accuracy = accuracy_score(y_te, y_du_doan)

print("\nKET QUA MANHATTAN")
print("k tot nhat:", k_tot)
print(f"Accuracy tap kiem tra: {accuracy:.4f}")

# Moc tham chieu Euclid theo tai lieu
print("\nSO SANH VOI EUCLID")
print("Euclid: k = 13, accuracy = 0.9200")
print(
    "Manhattan co accuracy gan moc Euclid:",
    abs(accuracy - 0.9200) <= 0.02
)

#Nhận xét: So sánh k tốt nhất và accuracy thực tế mà chương trình in ra với kết quả Euclid. 
#Hai cách đo có thể cho kết quả khác nhau vì Manhattan tính tổng trị tuyệt đối chênh lệch theo từng đặc trưng, 
#còn Euclid tính căn tổng bình phương chênh lệch. Không nên ghi trước kết quả Manhattan khi chưa chạy code.