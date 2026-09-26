import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/gia_nha.csv")
plt.scatter(df["so_phong"], df["gia"])
plt.xlabel("Số phòng")
plt.ylabel("Giá (tỷ đồng)")
plt.title("Mối quan hệ giữa số phòng và giá nhà")
plt.savefig("bai2.png", dpi=150)
plt.show()
print("Nhận xét: Số phòng càng tăng thì giá nhà có xu hướng tăng.")