
import time
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier


rng = np.random.default_rng(0)

kich_thuoc = [1000, 10000, 100000]
so_mau_moi = 1000

# Tao bo du lieu kiem tra co dinh de so sanh cong bang
X_moi = rng.random((so_mau_moi, 2))

ket_qua = []

print("BAI 6: DO THOI GIAN FIT VA PREDICT")
print("Moi bo du lieu co 2 dac trung.")
print("Dang do thoi gian, vui long cho...")
print()

for n in kich_thuoc:
    # Tao du lieu huan luyen ngau nhien
    X_hoc = rng.random((n, 2))

    # Tao nhan nhi phan tu du lieu
    y_hoc = (X_hoc[:, 0] + X_hoc[:, 1] > 1).astype(int)

    mo_hinh = KNeighborsClassifier(n_neighbors=5)

    # Do thoi gian fit
    bat_dau = time.perf_counter()
    mo_hinh.fit(X_hoc, y_hoc)
    thoi_gian_fit = time.perf_counter() - bat_dau

    # Do thoi gian predict cho 1000 mau moi
    bat_dau = time.perf_counter()
    du_doan = mo_hinh.predict(X_moi)
    thoi_gian_predict = time.perf_counter() - bat_dau

    ket_qua.append({
        "So mau hoc": n,
        "Thoi gian fit (giay)": thoi_gian_fit,
        "Thoi gian predict (giay)": thoi_gian_predict
    })

# In bang ket qua
bang = pd.DataFrame(ket_qua)

print("BANG THOI GIAN")
print(bang.to_string(
    index=False,
    formatters={
        "Thoi gian fit (giay)": lambda x: f"{x:.6f}",
        "Thoi gian predict (giay)": lambda x: f"{x:.6f}"
    }
))

print("\nNHAN XET")
print("KNN co thoi gian fit ngan vi chu yeu luu du lieu.")
print("Khi predict, KNN phai tim lang gieng cho tung mau moi.")
print("So mau hoc cang tang thi predict thuong cang ton thoi gian.")
print("Thoi gian cu the phu thuoc vao may va moi truong chay.")
#Nhận xét: KNN được gọi là mô hình học lười (lazy learning) vì quá trình huấn luyện chủ yếu lưu trữ dữ liệu, 
#không xây dựng một mô hình tham số phức tạp. Khi dự đoán, mô hình mới tính khoảng cách và tìm láng giềng gần nhất. 
#Vì vậy, thời gian huấn luyện thường ngắn nhưng thời gian dự đoán có thể tăng đáng kể khi số mẫu học tăng.