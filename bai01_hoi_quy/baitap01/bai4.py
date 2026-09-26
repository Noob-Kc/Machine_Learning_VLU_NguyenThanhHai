import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")

y = df["gia"]

# Mô hình một biến
X_mot = df[["dien_tich"]]

# Bài 4: hai biến đầu vào
X_ba = df[["dien_tich", "so_phong"]]

# Lưu R2 để so sánh
r2_mot = 0
r2_hai = 0

for ten, X in [("Mot bien ", X_mot), ("Hai bien ", X_ba)]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mo_hinh = LinearRegression()
    mo_hinh.fit(X_train, y_train)

    r2 = r2_score(y_test, mo_hinh.predict(X_test))

    print(f"{ten}: R2 tren tap kiem tra = {r2:.4f}")

    if ten == "Mot bien ":
        r2_mot = r2
    else:
        r2_hai = r2

print()
print("So sanh voi R2 = 0.9622 cua mo hinh mot bien trong tai lieu:")

if r2_hai > 0.9622:
    print(f"R2 cua mo hinh hai bien ({r2_hai:.4f}) cao hon 0.9622.")
    print("Viec them cot so_phong giup mo hinh cai thien R2 tren tap kiem tra.")
elif r2_hai < 0.9622:
    print(f"R2 cua mo hinh hai bien ({r2_hai:.4f}) thap hon 0.9622.")
    print("Viec them cot so_phong khong giup mo hinh cai thien R2 tren tap kiem tra.")
else:
    print("R2 cua mo hinh hai bien bang 0.9622.")
    print("Viec them cot so_phong khong lam thay doi R2.")

print()
print("Nhan xet: R2 cao hon cho thay mo hinh giai thich duoc nhieu bien thien cua gia nha hon.")