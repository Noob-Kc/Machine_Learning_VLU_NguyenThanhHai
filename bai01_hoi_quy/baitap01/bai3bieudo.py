import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/gia_nha.csv")

plt.scatter(df["tuoi_nha"], df["gia"])

plt.xlabel("Tuổi nhà")
plt.ylabel("Giá (tỷ đồng)")
plt.title("Mối quan hệ giữa tuổi nhà và giá nhà")

plt.savefig("bai3.png", dpi=150)
plt.show()
# Các điểm dữ liệu không tập trung sát quanh một đường thẳng mà phân tán khá rộng. 
# Vì vậy, mặc dù hệ số w mang dấu âm và cho thấy xu hướng giá giảm khi tuổi nhà tăng, 
# chỉ dựa vào dấu của w thì chưa thể kết luận hai đại lượng có quan hệ chặt chẽ. 
# Cần sử dụng thêm các chỉ số đánh giá như R² hoặc hệ số tương quan để đánh giá mức độ quan hệ.