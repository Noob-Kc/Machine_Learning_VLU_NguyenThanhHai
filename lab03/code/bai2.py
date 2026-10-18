
"""Bai 2: Tim 5 tu nghieng ve tin rac va 5 tu nghieng ve tin thuong."""

import csv
import math
from collections import Counter

# Doc du lieu CSV
with open("data/tin_nhan.csv", "r", encoding="utf-8-sig", newline="") as f:
    df = list(csv.DictReader(f))

# Tach tu giong CountVectorizer(token_pattern=r"(?u)\S+")
tu_theo_tin = []
nhan = []

for dong in df:
    tu = dong["noi_dung"].lower().split()
    tu_theo_tin.append(tu)
    nhan.append(int(dong["la_rac"]))

# Tap tu vung
tu_vung = sorted(set(tu for ds_tu in tu_theo_tin for tu in ds_tu))
so_tu = len(tu_vung)
alpha = 1.0

# Dem tan suat tu trong tung lop
dem_tu = {0: Counter(), 1: Counter()}
tong_tu = {0: 0, 1: 0}

for ds_tu, lop in zip(tu_theo_tin, nhan):
    dem_tu[lop].update(ds_tu)
    tong_tu[lop] += len(ds_tu)

# Tinh log P(tu | lop) voi Laplace smoothing
chenh_lech = []

for tu in tu_vung:
    p_rac = (dem_tu[1][tu] + alpha) / (
        tong_tu[1] + alpha * so_tu
    )
    p_thuong = (dem_tu[0][tu] + alpha) / (
        tong_tu[0] + alpha * so_tu
    )

    delta = math.log(p_rac) - math.log(p_thuong)
    chenh_lech.append((tu, delta))

# In 5 tu nghiêng ve tin rac
chenh_lech.sort(key=lambda x: (-x[1], x[0]))

print("5 tu to cao tin rac manh nhat:")
for tu, delta in chenh_lech[:5]:
    print(f"{tu:15s} {delta:.4f}")

print()

# In 5 tu nghiêng ve tin thuong
chenh_lech.sort(key=lambda x: (x[1], x[0]))

print("5 tu nghieng ve tin thuong manh nhat:")
for tu, delta in chenh_lech[:5]:
    print(f"{tu:15s} {delta:.4f}")
