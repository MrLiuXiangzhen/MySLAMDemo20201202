import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

img1 = cv2.imread('./image/1.jpg')  # 读取原图片
img2 = cv2.imread('./image/2.jpg')  # 读取旋转图片
img3 = cv2.imread('./image/3.jpg')  # 读取缩放图片

# 构建仿射图
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

# 使用cv2.imread()接口读图像，读进来的是BGR格式以及[0～255]。所以要将img转换为RGB格式，不然后面显示会有色差
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
gray4 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# sift 特征提取
a = time.time()
sift1 = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift1.detectAndCompute(img1, None)
kp2, des2 = sift1.detectAndCompute(img2, None)
kp3, des3 = sift1.detectAndCompute(img3, None)
kp4, des4 = sift1.detectAndCompute(img4, None)
b = time.time()
print("sift提取时间为：", b - a)

# 原图特征提取展示
img11 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img11)
plt.title('Left:yuan---Right:yuan')
plt.axis('off')
plt.show()
# 旋转图特征提取展示
img12 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img12)
plt.title('Left:yuan---Right:xuanzhuan')
plt.axis('off')
plt.show()
# 缩放图特征提取展示
img13 = cv2.drawKeypoints(img3, kp3, img3, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img13)
plt.title('Left:yuan---Right:suolue')
plt.axis('off')
plt.show()
# 仿射图特征提取展示
img14 = cv2.drawKeypoints(img4, kp4, img4, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img14)
plt.title('Left:yuan---Right:fangshe')
plt.axis('off')
plt.show()

# SIFT特征匹配
# FLANN 参数设计
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 与原图的匹配连线效果图展示
matches = flann.knnMatch(des1, des1, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
img41 = cv2.drawMatchesKnn(img1, kp1, img1, kp1, good, None, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img41)
plt.title('Left:yuan---Right:yuan')
plt.axis('off')
plt.show()

# 与旋转图的匹配连线效果图展示
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
img42 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img42)
plt.title('Left:yuan---Right:xuanzhuan')
plt.axis('off')
plt.show()

# 原图与缩略图的匹配连线效果图展示
matches = flann.knnMatch(des1, des3, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
img43 = cv2.drawMatchesKnn(img1, kp1, img3, kp3, good, None, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img43)
plt.title('Left:yuan---Right:suolue')
plt.axis('off')
plt.show()

# 与仿射图的匹配结果显示
matches = flann.knnMatch(des1, des4, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
img44 = cv2.drawMatchesKnn(img1, kp1, img4, kp4, good, None, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img44)
plt.title('Left:yuan---Right:fangshe')
plt.axis('off')
plt.show()
