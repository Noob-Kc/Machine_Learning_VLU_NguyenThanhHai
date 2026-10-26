
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Chia du lieu giong tai lieu bai thuc hanh
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Tim k tot nhat bang kiem dinh cheo 5 phan
diem = {}

print("BAI 5: CHON K CHO HOI QUY")
print("Diem kiem dinh cheo (MAE, don vi ty dong)")
print("-" * 45)

for k in range(1, 21):
    mo_hinh = KNeighborsRegressor(n_neighbors=k)

    diem_am = cross_val_score(
        mo_hinh,
        X_tr,
        y_tr,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    # Chuyen diem am thanh MAE duong
    mae_cv = -diem_am.mean()
    diem[k] = mae_cv

    print(f"k = {k:2d}: MAE CV = {mae_cv:.4f}")

# MAE nho nhat la tot nhat
k_tot = min(diem, key=diem.get)

# Huan luyen KNN voi k tot nhat
knn = KNeighborsRegressor(n_neighbors=k_tot)
knn.fit(X_tr, y_tr)
du_doan_knn = knn.predict(X_te)
mae_knn = mean_absolute_error(y_te, du_doan_knn)

# Huan luyen hoi quy tuyen tinh de so sanh
linear = LinearRegression()
linear.fit(X_tr, y_tr)
du_doan_linear = linear.predict(X_te)
mae_linear = mean_absolute_error(y_te, du_doan_linear)

print("\nKET QUA CUOI CUNG")
print("k tot nhat:", k_tot)
print(f"MAE CV tot nhat: {diem[k_tot]:.4f}")
print(f"MAE KNN tren tap kiem tra: {mae_knn:.4f}")
print(f"MAE hoi quy tuyen tinh:    {mae_linear:.4f}")

if mae_knn < mae_linear:
    print("Ket luan: KNN tot hon hoi quy tuyen tinh.")
elif mae_knn > mae_linear:
    print("Ket luan: Hoi quy tuyen tinh tot hon KNN.")
else:
    print("Ket luan: Hai mo hinh co MAE bang nhau.")
#Giải thích: 
#cross_val_score trả điểm âm vì dùng tiêu chí neg_mean_absolute_error. 
#Do đó, phải đổi dấu bằng -diem_am.mean() để thu được MAE dương. 
#Chọn k có MAE kiểm định chéo nhỏ nhất rồi mới đánh giá trên tập kiểm tra.

#Ở ví dụ trong tài liệu, hồi quy tuyến tính có MAE 0.2578 tỷ đồng, trong khi KNN với k=5 có MAE 0.4030 tỷ đồng. Lab04_KNN.pdf
#Tuy nhiên, kết quả của bài này cần dùng k được chọn từ kiểm định chéo, không mặc định là 5.