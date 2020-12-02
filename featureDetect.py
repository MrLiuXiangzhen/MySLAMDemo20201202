"""
特征提取与匹配
kds==> [[keypoints, descriptors], [keypoints, descriptors]]
matches==>  queryIdx：查询图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
            trainIdx：训练图像的特征点描述符下标,同时也是描述符对应特征点的下标。
            distance：代表匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
            对于keypoints[matches.queryIdx] --> .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小
"""
import numpy as np
import matplotlib.pyplot as mp
import cv2
import time


class FeatureDetect(object):
    def __init__(self, method='orb', draw=False):
        '''
        对输入的两张图像路径里的图片进行 特征点绘制并匹配
        :param method: orb or sift
        :draw   if False run() return matchs
              elif True  run() return RGB图像列表，绘制好的关键点图像，特征匹配连线后的图像
        '''
        self.method = method
        self.draw = draw

    @staticmethod
    def imgRead(image):
        '''
        :param image: 图像路径
        :return: 读取并转换好的RGB图像
        '''
        img = cv2.imread(image)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 特征提取
    def detectAndCompute(self, img):
        '''
        :param img: 原图片
        :return: [关键点， 描述子] , keypoints, descriptors
        '''
        if self.method == 'orb':
            orb = cv2.ORB_create()
            return orb.detectAndCompute(img, None)  # 返回值：关键点和描述子
        elif self.method == 'sift':
            sift = cv2.xfeatures2d.SIFT_create()
            return sift.detectAndCompute(img, None)

    # 特征点匹配
    def match(self, kds):
        '''
        :param kds: [[keypoints, descriptors], [keypoints, descriptors]]
        :return: 绘制好的特征点连线图
        '''
        if self.method == 'sift':
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            # 与原图的匹配连线效果图展示
            matches = flann.knnMatch(kds[0][1], kds[1][1], k=2)
            # matchesMask = [[0, 0] for i in range(len(matches))]
            matche = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    matche.append([m])
            return matche
        else:
            # BFMatcher：一种二维特征点匹配方法，暴力法
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)

            # orb的normType应该使用NORM_HAMMING，筛选的依据是汉明距离小于最小距离
            matches = bf.match(kds[0][1], kds[1][1])  # 对图像中的BRIEF描述子进行匹配
            # matches = sorted(matches, key=lambda x: x.distance)  # 根据距离排序

            # matches = flann.knnMatch(kds[0][1], kds[1][1], k=2)  # drawMatchesKnn，match()匹配点对，knnMatch()返回每个点的k个匹配点

            for m in matches:
                for n in matches:
                    if (m != n and m.distance >= n.distance * 0.7):
                        matches.remove(m)
                        break
            return matches

    # 绘制关键点
    def drawKeyPoint(self, imgs, kds):
        '''
        :param imgs: 图像列表
        :param kds: [[keypoints, descriptors], [keypoints, descriptors]]
        :return: 绘制好关键点的图像
        '''
        img_ps = []
        for num, kd in enumerate(kds):
            img_ps.append(cv2.drawKeypoints(imgs[num], kd[0], imgs[num], color=(255, 0, 255)))
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
        return img_ps

    def run(self, images):
        '''
        :param images: 图像路径列表
        :return: RGB图像列表，绘制好的关键点图像，特征匹配连线后的图像 [pics, imgKeyPoints, imgFeature] or [kds, matches]
        '''
        imgs = []

        # 读取图片并转换RGB
        for i in images:
            imgs.append(self.imgRead(i))
        kds = []

        # 特征提取，获取关键点和描述子
        start = time.time()
        for img in imgs:
            keypoints, descriptors = self.detectAndCompute(img)
            kds.append([keypoints, descriptors])
        end = time.time()
        print("提取时间为：", end - start)

        # 绘制关键点
        img_ps = self.drawKeyPoint(imgs, kds)

        # 特征点匹配
        matches = self.match(kds)
        # print(kds[0][0][matches[0].queryIdx].pt)
        # print(kds[1][0][matches[0].trainIdx].pt)

        if self.draw:
            # 连线
            if self.method == 'sift':
                img_f = cv2.drawMatchesKnn(imgs[0], kds[0][0], imgs[1], kds[1][0], matches[:50], None, flags=2)
            else:  # orb特征
                img_f = cv2.drawMatches(imgs[0], kds[0][0], imgs[1], kds[1][0], matches[:50], imgs[1], flags=2)
            '''
               cv2.drawMatches()其中参数如下：
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
            return imgs, img_ps, img_f
        else:
            return kds, matches
    """
    queryIdx：查询图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    trainIdx：训练图像的特征点描述符下标,同时也是描述符对应特征点的下标。
    distance：代表匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小
    """


if __name__ == '__main__':
    fd = FeatureDetect('orb', draw=True)
    pics, imgKeyPoints, imgFeature = fd.run(['./image/1.jpg', './image/2.jpg'])  # draw 为True
    # kds, matches = fd.run(['./pics/1.png', './pics/2.png'])  # draw 为False
    # print(kds[1][0])
    # print(type(matches[0]))
    # print(matches[0][0].queryIdx)

    # 绘图
    mp.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mp.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    mp.figure('特征点提取')

    mp.subplot(221)
    mp.imshow(pics[0])
    mp.title('原图1')
    mp.axis('off')

    mp.subplot(222)
    mp.imshow(pics[1])
    mp.title('原图2')
    mp.axis('off')

    mp.subplot(223)
    mp.imshow(imgKeyPoints[0])
    mp.title('图1关键点')
    mp.axis('off')

    mp.subplot(224)
    mp.imshow(imgKeyPoints[1])
    mp.title('图2关键点')
    mp.axis('off')

    mp.figure('特征匹配')
    mp.imshow(imgFeature)
    mp.title('特征匹配')
    mp.axis('off')
    mp.show()
