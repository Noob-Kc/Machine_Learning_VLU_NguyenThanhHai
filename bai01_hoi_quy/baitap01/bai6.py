# Hệ số của mô hình một biến
w = 0.078367
b = 0.401752


def du_doan_gia(dien_tich):
    # Cảnh báo nếu diện tích nằm ngoài dữ liệu đã học
    if dien_tich < 35.5 or dien_tich > 117.5:
        print("Cảnh báo: Diện tích nằm ngoài khoảng dữ liệu đã học.")

    # Tính giá dự đoán
    gia_du_doan = w * dien_tich + b

    return gia_du_doan


# Gọi hàm với 3 diện tích
gia_60 = du_doan_gia(60)
gia_80 = du_doan_gia(80)
gia_200 = du_doan_gia(200)

print(f"Giá dự đoán cho căn 60 m²: {gia_60:.3f} tỷ đồng")
print(f"Giá dự đoán cho căn 80 m²: {gia_80:.3f} tỷ đồng")
print(f"Giá dự đoán cho căn 200 m²: {gia_200:.3f} tỷ đồng")