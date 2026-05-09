import cv2
import numpy as np


total = 0  # 初始化總金額為 0
count = {"50": 0, "10": 0, "5": 0, "1": 0}


# 硬幣判斷
def coinPrice(r: int) -> int:
    """
    根據硬幣的半徑判斷其面值。

    Args:
        r (int): 硬幣的半徑。

    Returns:
        int: 硬幣的面值（以元計）。
    """
    price = 0
    if r > 55:  # 半徑大於 55 的硬幣面值為 50 元
        price = 50
        count["50"] += 1
    elif r < 55 and r > 49:  # 半徑介於 50 和 55 之間的硬幣面值為 10 元
        price = 10
        count["10"] += 1
    elif r < 49 and r > 42:  # 半徑介於 43 和 49 之間的硬幣面值為 5 元
        price = 5
        count["5"] += 1
    elif r < 42 and r > 30:  # 半徑介於 31 和 42 之間的硬幣面值為 1 元
        price = 1
        count["1"] += 1
    return price


# 讀取影像
coin = cv2.imread("coin.jpg")  # 讀取名為 coin.jpg 的影像檔
coin = cv2.resize(coin, (500, 500))  # 將影像調整為 500x500 大小
gray_coin = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)  # 將影像轉換為灰階影像
cv2.imwrite("gray.jpg", gray_coin)  # 將模糊後的影像存為 gray_coin.jpg

# 中值濾波
median = cv2.medianBlur(gray_coin, 17)  # 對灰階影像進行中值模糊，模糊程度為 17
cv2.imwrite("median.jpg", median)  # 將模糊後的影像存為 median.jpg

# 邊緣檢測
Laplacian = cv2.Laplacian(
    median,  # 輸入影像
    ddepth=-1,  # 輸出影像深度（與輸入影像一致）
    ksize=1,  # 核大小（使用 1x1 的濾波器）
    scale=1,  # 邊緣檢測的比例因子
)
cv2.imwrite("Laplacian.jpg", Laplacian)  # 將邊緣檢測結果存為 Laplacian.jpg

# 霍夫圓變換檢測圓形
Hough = cv2.HoughCircles(
    Laplacian,  # 邊緣檢測後的影像
    cv2.HOUGH_GRADIENT,  # 使用霍夫梯度法進行圓檢測
    dp=1.2,  # 累積器解析度與輸入影像解析度的比例
    minDist=30,  # 圓心之間的最小距離
    param1=13,  # 邊緣檢測的高閾值
    param2=55,  # 檢測圓的累積器閾值
    minRadius=30,  # 圓半徑的最小值
    maxRadius=100,  # 圓半徑的最大值
)

# 如果檢測到圓形
if Hough is not None:
    Hough = np.round(Hough[0, :]).astype("int")  # 將檢測到的圓的參數取整數
    for x, y, r in Hough:  # 遍歷每個檢測到的圓
        cv2.circle(
            coin, (x, y), r, (0, 0, 255), 2
        )  # 在原圖上畫出圓形，顏色為紅色，線寬為 2
        total += coinPrice(r)  # 根據圓的半徑計算面值並累加到總金額

# 在圖片上顯示總金額
cv2.putText(
    img=coin,  # 輸入影像
    text=f"{total} dollars",  # 要顯示的文字內容
    org=(5, 40),  # 文字位置（左上角）
    fontFace=5,  # 字體類型
    fontScale=1.5,  # 字體大小
    color=(255, 0, 255),  # 字體顏色（紫色）
    thickness=2,  # 字體線條厚度
    lineType=cv2.LINE_AA,  # 抗鋸齒線條
)

# 輸出結果
print('硬幣個數')
print(f"50元 : {count['50']} 個\n10元 : {count['10']} 個\n\
5元 : {count['5']} 個\n1元 : {count['1']} 個\n")
print(f"圖片中共 {total} 元")
cv2.imwrite("output.jpg", coin)  # 將結果存為 output.jpg
