"""Bai 1: Bayes voi ba nguyen nhan."""
# Ty le tung nhom
prior = [0.50, 0.30, 0.20]
# Xac suat qua mon cua tung nhom
p_qua = [0.95, 0.70, 0.30]
ten_nhom = ["Hoc deu", "Hoc don", "It hoc"]

# P(qua mon) theo cong thuc xac suat toan phan
p_tong_qua = sum(prior[i] * p_qua[i] for i in range(3))

print(f"P(qua mon) = {p_tong_qua:.4f}")
print("Xac suat thuoc tung nhom khi da biet ban do qua mon:")
tong_hau_nghiem = 0
for i in range(3):
    p_nhom = prior[i] * p_qua[i] / p_tong_qua
    tong_hau_nghiem += p_nhom
    print(f"{ten_nhom[i]}: {p_nhom:.4f} ({p_nhom * 100:.2f}%)")
print(f"Tong ba xac suat = {tong_hau_nghiem:.4f}")
