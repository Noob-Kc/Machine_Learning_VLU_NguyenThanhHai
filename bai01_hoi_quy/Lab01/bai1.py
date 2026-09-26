import pandas as pd
#cd /d C:\hocmayvaungdung\bai01_hoi_quy
#python Lab01\bai1.py
df = pd.read_csv("data/gia_nha.csv")
nhom_lon = df[df["dien_tich"] > 100]
print(len(nhom_lon))
print(nhom_lon["gia"].mean())