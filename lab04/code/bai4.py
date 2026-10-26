
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

print("BAI 4: KNN CO TRONG SO")
print("So sanh weights='uniform' va weights='distance'")
print("-" * 66)
print(f"{'k':>5} {'Uniform':>15} {'Distance':>15}")
print("-" * 66)

for k in [1, 13, 101]:
    # KNN bo phieu deu
    knn_deu = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform"
    )
    knn_deu.fit(X_tr, y_tr)
    acc_deu = accuracy_score(
        y_te, knn_deu.predict(X_te)
    )

    # KNN co trong so theo khoang cach
    knn_trong_so = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )
    knn_trong_so.fit(X_tr, y_tr)
    acc_trong_so = accuracy_score(
        y_te, knn_trong_so.predict(X_te)
    )

    print(f"{k:5d} {acc_deu:15.4f} {acc_trong_so:15.4f}")

print("\nNHAN XET")
print("weights='distance' uu tien cac lang gieng gan hon.")
print("Voi k=1, ket qua hai cach tinh trong so thuong giong nhau.")
print("Voi k=101, trong so khoang cach co the giam anh huong")
print("cua cac diem xa, nhung khong dam bao accuracy tang.")
#Khi weights="distance", các điểm gần mẫu cần dự đoán có ảnh hưởng lớn hơn các điểm xa. 
#Với k=101, cách này có thể giúp giảm ảnh hưởng của những láng giềng xa, nhưng không đảm bảo mô hình tốt hơn trong mọi trường hợp.
#Cần dựa vào bảng kết quả thực tế để kết luận.