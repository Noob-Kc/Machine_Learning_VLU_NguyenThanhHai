## toc_do_hoc = 0.001
# "C:\hocmayvaungdung\bai01_hoi_quy>python baitap01\b6_gradient_descent.py
# Vong w b MSE
#   1  0.0031  0.0130  44.6329
#   2  0.0063  0.0259  44.4553
#   5  0.0156  0.0647  43.9266
#  10  0.0311  0.1287  43.0595
#  25  0.0767  0.3170  40.5599
#  50  0.1497  0.6185  36.7135
# 100  0.2851  1.1781  30.0849
# 200  0.5184  2.1425  20.2175

# Sau khi doi ve thang do met vuong:
#  w = 0.025857
#  b = 0.132557
# So voi cong thuc o buoc 3: w = 0.078367, b = 0.401752
print("------------------------------")
## toc_do_hoc = 1.02
# C:\hocmayvaungdung\bai01_hoi_quy>python baitap01\b6_gradient_descent.py
# Vong w b MSE
#   1  3.2054 13.2464  48.4532
#   2 -0.1282 -0.5299  52.3924
#   5  3.4829 14.3935  66.2456
#  10 -0.7546 -3.1184  97.9737
#  25  5.7600 23.8035 317.3662
#  50 -9.5952 -39.6527 2254.3276
# 100 -77.7851 -321.4521 113845.8345
# 200 -4006.3165 -16556.3753 290391782.0192

# Sau khi doi ve thang do met vuong:
#  w = -199.815704
#  b = -1024.367599
# So voi cong thuc o buoc 3: w = 0.078367, b = 0.401752
print("------------------------------")
## Với tốc độ học 0.001, MSE tại vòng lặp thứ 200 là 20.2175. 
# Do tốc độ học nhỏ nên mỗi bước cập nhật rất ngắn, thuật toán giảm sai số chậm và sau 200 vòng lặp vẫn chưa tiến gần đến giá trị tối ưu 0.1790.

# Với tốc độ học 1.02, MSE tại vòng lặp thứ 200 tăng lên rất lớn, đạt 290391782.0192. 
# Nguyên nhân là tốc độ học quá lớn, mỗi bước cập nhật quá dài khiến thuật toán liên tục vượt qua điểm cực tiểu 
# và đi xa khỏi nghiệm tốt, làm MSE tăng mạnh.

# Qua hai trường hợp trên có thể thấy tốc độ học cần được lựa chọn phù hợp. 
# Tốc độ học quá nhỏ khiến thuật toán hội tụ chậm, còn tốc độ học quá lớn khiến thuật toán không hội tụ.

print("---------------------------------------------------------------------------")
## toc_do_hoc = 0.5
#C:\hocmayvaungdung\bai01_hoi_quy>python baitap01\b6_gradient_descent.py
#Vong w b MSE
#   1  1.5713  6.4933   0.1790
#   2  1.5713  6.4933   0.1790
#   5  1.5713  6.4933   0.1790
#  10  1.5713  6.4933   0.1790
#  25  1.5713  6.4933   0.1790
#  50  1.5713  6.4933   0.1790
# 100  1.5713  6.4933   0.1790
# 200  1.5713  6.4933   0.1790

#Sau khi doi ve thang do met vuong:
# w = 0.078367
# b = 0.401752

#So voi cong thuc o buoc 3: w = 0.078367, b = 0.401752

## Giải thích
#Với tốc độ học 0.5, MSE giảm xuống 0.1790 ngay từ vòng lặp đầu tiên và giữ ổn định đến vòng 200. 
#Sau khi đưa về thang đo ban đầu, hệ số w = 0.078367, b = 0.401752, trùng với kết quả tính bằng công thức ở bước 3. 
#Điều này cho thấy tốc độ học 0.5 giúp Gradient Descent hội tụ rất nhanh và ổn định
print("------------------------------")
## toc_do_hoc = 0.9
#C:\hocmayvaungdung\bai01_hoi_quy>python baitap01\b6_gradient_descent.py
#Vong w b MSE
#   1  2.8283 11.6880  28.7437
#   2  0.5657  2.3376  18.4604
#   5  2.0861  8.6211   4.9714
#  10  1.4025  5.7961   0.6936
#  25  1.5772  6.5179   0.1797
#  50  1.5712  6.4932   0.1790
# 100  1.5713  6.4933   0.1790
# 200  1.5713  6.4933   0.1790

#Sau khi doi ve thang do met vuong:
# w = 0.078367
# b = 0.401752
#So voi cong thuc o buoc 3: w = 0.078367, b = 0.401752

## Giải thích
#Với tốc độ học 0.9, MSE ban đầu còn khá lớn nhưng giảm nhanh qua các vòng lặp. 
#Đến khoảng vòng 25, MSE đạt gần giá trị tối ưu 0.1790 và từ vòng 50 trở đi gần như không thay đổi. 
#Như vậy, tốc độ học 0.9 vẫn giúp thuật toán hội tụ ổn định, nhưng có dao động lớn ở những vòng lặp đầu.