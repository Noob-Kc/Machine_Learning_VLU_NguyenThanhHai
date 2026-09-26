import pandas as pd
# Đọc dữ liệu
df = pd.read_csv("data/gia_nha.csv")
# Biến đầu vào: tuổi nhà
x = df["tuoi_nha"].values
# Biến cần dự đoán: giá
y = df["gia"].values
# Tính hệ số góc w
w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
# Tính hệ số chặn b
b = y.mean() - w * x.mean()
print(f"w = {w:.6f}")
print(f"b = {b:.6f}")
# Nhận xét
if w < 0:
    print("Nhận xét: w mang dấu âm.")
    print("Điều này có nghĩa là khi tuổi nhà tăng thêm 1 năm thì giá nhà có xu hướng giảm.")
else:
    print("Nhận xét: w mang dấu dương.")
    print("Điều này có nghĩa là khi tuổi nhà tăng thêm 1 năm thì giá nhà có xu hướng tăng.")