import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

img10 = cv2.imread('./image/1.jpg')  # 读取原图片
img20 = cv2.imread('./image/2.jpg')  # 读取旋转图片
img30 = cv2.imread('./image/3.jpg')  # 读取缩放图片

# 构建放射图
img = cv2.cvtColor(img10, cv2.COLOR_BGR2RGB)
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

# 使用cv2.imread()接口读图像，读进来的是BGR格式以及[0～255]。所以要将img转换为RGB格式，不然后面显示会有色差
img1 = cv2.cvtColor(img10, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
img2 = cv2.cvtColor(img20, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img20, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img30, cv2.COLOR_BGR2RGB)
gray3 = cv2.cvtColor(img30, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
gray4 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# ORB 特征提取
a = time.time()
sift2 = cv2.ORB_create()
kp1, des1 = sift2.detectAndCompute(img1, None)  # 返回值：关键点和描述子
"""
kp关键点包含的信息：
angle：角度，表示关键点的方向，通过Lowe大神的论文可以知道，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
octave：代表是从金字塔哪一层提取的得到的数据。
pt：关键点点的坐标
response：响应程度，代表该点强壮大小，更确切的说，是该点角点的程度。
size：该点直径的大小
"""
kp2, des2 = sift2.detectAndCompute(img2, None)
kp3, des3 = sift2.detectAndCompute(img3, None)
kp4, des4 = sift2.detectAndCompute(img4, None)
b = time.time()
print("ORB提取时间为：", b - a)

# 对原图特征提取
img11 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
'''
第一个参数image：原始图像，可以使三通道或单通道图像；
第二个参数keypoints：特征点向量，向量内每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息；
第三个参数outImage：特征点绘制的画布图像，可以是原图像；
第四个参数color：绘制的特征点的颜色信息，默认绘制的是随机彩色；
第五个参数flags：特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制，有以下几种模式可选：
　　DEFAULT：只绘制特征点的坐标点,显示在图像上就是一个个小圆点,每个小圆点的圆心坐标都是特征点的坐标。
　　DRAW_OVER_OUTIMG：函数不创建输出的图像,而是直接在输出图像变量空间绘制,要求本身输出图像变量就 是一个初始化好了的,size与type都是已经初始化好的变量
　　NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制
　　DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆,这种方法同时显示图像的坐 标,size，和方向,是最能显示特征的一种绘制方式
'''
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img11)
plt.title('Left:yuan---Right:yuan')
plt.axis('off')
plt.show()

# 对旋转图特征提取
img12 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img12)
plt.title('Left:yuan---Right:xuanzhuan')
plt.axis('off')
plt.show()

# 对缩放图特征提取
img13 = cv2.drawKeypoints(img3, kp3, img3, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img13)
plt.title('Left:yuan---Right:suolue')
plt.axis('off')
plt.show()

# 对仿射图特征提取
img14 = cv2.drawKeypoints(img4, kp4, img4, color=(255, 0, 255))
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img14)
plt.title('Left:yuan---Right:fangshe')
plt.axis('off')
plt.show()

# 与原图的匹配连线效果图展示
# BFMatcher：一种二维特征点匹配方法，暴力法
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# orb的normType应该使用NORM_HAMMING，筛选的依据是汉明距离小于最小距离
matches = bf.match(des1, des1)  # 对图像中的BRIEF描述子进行匹配
matches = sorted(matches, key=lambda x: x.distance)  # 根据距离排序
knnMatches = bf.knnMatch(des1, des1, k=1)  # drawMatchesKnn，match()匹配点对，knnMatch()返回每个点的k个匹配点
for m in matches:
    for n in matches:
        if (m != n and m.distance >= n.distance * 0.75):
            matches.remove(m)
            break
img = cv2.drawMatches(img1, kp1, img1, kp1, matches[:50], img1, flags=2)
'''
其中参数如下：
img1 – 源图像1
keypoints1 –源图像1的特征点
img2 – 源图像2
keypoints2 – 源图像2的特征点
matches1to2 – 源图像1的特征点匹配源图像2的特征点[matches[i]]
outImg – 输出图像具体由flags决定
matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1)，颜色随机
singlePointColor – 单个点的颜色，即未配对的特征点，若matchColor==Scalar::all(-1)，颜色随机
matchesMask – Mask决定哪些点将被画出，若为空，则画出所有匹配点
flags – Fdefined by DrawMatchesFlags
'''
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img)
plt.title('Left:yuan---Right:yuan')
plt.axis('off')
plt.show()

# 原图与旋转图的匹配连线效果图展示
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # orb的normType应该使用NORM_HAMMING
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
for m in matches:
    for n in matches:
        if (m != n and m.distance >= n.distance * 0.75):
            matches.remove(m)
            break
img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img)
plt.title('Left:yuan---Right:xuanzhuan')
plt.axis('off')
plt.show()

# 原图与缩略图的匹配连线效果图展示
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # orb的normType应该使用NORM_HAMMING
matches = bf.match(des1, des3)
matches = sorted(matches, key=lambda x: x.distance)
knnMatches = bf.knnMatch(des1, des3, k=1)  # drawMatchesKnn
for m in matches:
    for n in matches:
        if (m != n and m.distance >= n.distance * 0.75):
            matches.remove(m)
            break
img = cv2.drawMatches(img1, kp1, img3, kp3, matches[:50], img3, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img)
plt.title('Left:yuan---Right:suolue')
plt.axis('off')
plt.show()

# 原图与仿射图的匹配连线效果图展示
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # orb的normType应该使用NORM_HAMMING
matches = bf.match(des1, des4)
matches = sorted(matches, key=lambda x: x.distance)
knnMatches = bf.knnMatch(des1, des4, k=1)  # drawMatchesKnn
for m in matches:
    for n in matches:
        if (m != n and m.distance >= n.distance * 0.75):
            matches.remove(m)
            break
img = cv2.drawMatches(img1, kp1, img4, kp4, matches[:50], img4, flags=2)
plt.figure(num=1, figsize=(16, 16))
plt.imshow(img)
plt.title('Left:yuan---Right:fangshe')
plt.axis('off')
plt.show()
