
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/sinh_vien_mo_rong.csv")
y = df["qua_mon"]

# Truong hop 1: Chi tieu tinh bang nghin dong
cot_goc = ["gio_on", "diem_giua_ky", "chi_tieu"]
X_goc = df[cot_goc].copy()

# Truong hop 2: Doi chi tieu sang trieu dong
df["chi_tieu"] = df["chi_tieu"] / 1000
cot_moi = ["gio_on", "diem_giua_ky", "chi_tieu"]
X_moi = df[cot_moi].copy()

X_tr_goc, X_te_goc, y_tr, y_te = train_test_split(
    X_goc, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

X_tr_moi, X_te_moi, _, _ = train_test_split(
    X_moi, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# KNN khong chuan hoa truoc khi doi don vi
knn_goc = KNeighborsClassifier(n_neighbors=7)
knn_goc.fit(X_tr_goc, y_tr)
acc_goc = accuracy_score(
    y_te, knn_goc.predict(X_te_goc)
)

# KNN khong chuan hoa sau khi doi don vi
knn_moi = KNeighborsClassifier(n_neighbors=7)
knn_moi.fit(X_tr_moi, y_tr)
acc_moi = accuracy_score(
    y_te, knn_moi.predict(X_te_moi)
)

# KNN co chuan hoa truoc khi doi don vi
knn_chuan_goc = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7)
)
knn_chuan_goc.fit(X_tr_goc, y_tr)
acc_chuan_goc = accuracy_score(
    y_te, knn_chuan_goc.predict(X_te_goc)
)

# KNN co chuan hoa sau khi doi don vi
knn_chuan_moi = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7)
)
knn_chuan_moi.fit(X_tr_moi, y_tr)
acc_chuan_moi = accuracy_score(
    y_te, knn_chuan_moi.predict(X_te_moi)
)

print("BAI 3: ANH HUONG CUA DON VI DO")
print(f"Khong chuan hoa, don vi cu: {acc_goc:.4f}")
print(f"Khong chuan hoa, don vi moi: {acc_moi:.4f}")
print(f"Co chuan hoa, don vi cu: {acc_chuan_goc:.4f}")
print(f"Co chuan hoa, don vi moi: {acc_chuan_moi:.4f}")

print("\nNHAN XET")
print("Doi don vi co the lam thay doi khoang cach va ket qua KNN")
print("khi khong chuan hoa.")
print("Sau StandardScaler, ket qua thuong khong doi do chi")
print("tieu chi duoc doi don vi theo mot he so khac 0.")
#Khi không chuẩn hóa, cột chi tiêu có thể chi phối khoảng cách. 
#Chia cho 1000 làm thay đổi độ lớn của cột này nên các láng giềng được chọn có thể thay đổi.

#Khi dùng StandardScaler bên trong pipeline, phép đổi đơn vị tuyến tính này được loại bỏ bởi quá trình chuẩn hóa; 
#kết quả dự đoán về nguyên tắc không đổi, ngoại trừ các trường hợp sai số số học hoặc hòa khoảng cách.

#Accuracy cụ thể phải lấy từ lần chạy trên dữ liệu của bạn.