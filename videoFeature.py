import cv2
import matplotlib.pyplot as mp
import numpy as np
from threading import Thread
from time import sleep
import sys


class VideoFeature(object):
    def __init__(self, method='orb'):
        self.method = method
        self.pipe_imgs = []
        self.pipe_draw_points = []
        self.pipe_draw_matches = []
        self.pipe_kds = []
        self.matches = []

    def v_read(self, v_path):
        video_list = cv2.VideoCapture(v_path)
        s, image = video_list.read()
        while s:
            try:
                s, image = video_list.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.pipe_imgs.append(image)
                s, image = video_list.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.pipe_imgs.append(image)
                # cv2.waitKey(10)
                # print(image1)
            except Exception:
                s = False
                break

            # 特征提取，获取关键点和描述子
            keypoints, descriptors = self.detectAndCompute(self.pipe_imgs[0])
            self.pipe_kds.append([keypoints, descriptors])
            keypoints, descriptors = self.detectAndCompute(self.pipe_imgs[1])
            self.pipe_kds.append([keypoints, descriptors])

            # 绘制关键点
            img_ps = self.drawKeyPoint(self.pipe_imgs[0], self.pipe_kds[0][0])
            # print(img_ps)
            self.pipe_draw_points.append(img_ps)

            # 特征点匹配
            self.matches.append(self.match())
            # print(self.pipe_kds[0][0][self.matches[0].queryIdx].pt)
            # print(self.pipe_kds[1][0][self.matches[0].trainIdx].pt)

            # 连线
            self.pipe_draw_matches.append(self.drawMatch())
            # self.draw_img(img_f)
            # s = False
            self.pipe_imgs.pop(0)
            self.pipe_kds.pop(0)
            self.pipe_imgs.pop(0)
            self.pipe_kds.pop(0)

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
    def match(self):
        if self.method == 'sift':
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            # 与原图的匹配连线效果图展示
            matches = flann.knnMatch(self.pipe_kds[0][1], self.pipe_kds[1][1], k=2)
            matche = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    matche.append([m])
            return matche
        else:
            # BFMatcher：一种二维特征点匹配方法，暴力法
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # orb的normType应该使用NORM_HAMMING，筛选的依据是汉明距离小于最小距离
            matches = bf.match(self.pipe_kds[0][1], self.pipe_kds[1][1])  # 对图像中的BRIEF描述子进行匹配
            for m in matches:
                for n in matches:
                    if (m != n and m.distance >= n.distance * 0.7):
                        matches.remove(m)
                        break
            return matches

    # 绘制关键点
    def drawKeyPoint(self, img, keypoints):
        return cv2.drawKeypoints(img, keypoints, img, color=(255, 0, 255))

    # 绘制连线
    def drawMatch(self):
        if self.method == 'sift':
            img_m = cv2.drawMatchesKnn(self.pipe_imgs[0], self.pipe_kds[0][0], self.pipe_imgs[1], self.pipe_kds[1][0], self.matches[0][:50], None, flags=2)
        else:  # orb特征
            img_m = cv2.drawMatches(self.pipe_imgs[0], self.pipe_kds[0][0], self.pipe_imgs[1], self.pipe_kds[1][0], self.matches[0][:50], self.pipe_imgs[1], flags=2)
        return img_m

    def draw_img(self, img_f):
        mp.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
        mp.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        mp.figure('特征点提取')
        mp.imshow(img_f)
        mp.title('原图1')
        mp.axis('off')
        mp.show()

    def update(self, i):
        # print('update function')
        try:
            if len(self.pipe_draw_points):
                self.imshow.set_array(self.pipe_draw_points[0])
                self.pipe_draw_points.pop(0)
                return [self.imshow]
            else:
                sleep(0.1)
                return [self.imshow]
        except Exception as e:
            print(e)
            sys.exit('异常退出')

    def draw_update(self):
        mp.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
        mp.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        fig = mp.figure('Feature')
        self.imshow = mp.imshow(self.pipe_draw_points[0])
        # mp.axis('off')
        # mp.xlabel('xlabel', fontsize=16)
        mp.tight_layout()
        import matplotlib.animation as ma
        anim = ma.FuncAnimation(fig, self.update, frames=200, interval=30, blit=True)
        mp.show()

    def run(self, v_path):
        t = Thread(target=self.v_read, args=(0,))
        t.daemon = True
        t.start()
        sleep(5)
        # print(self.pipe_draw_points)
        self.draw_update()
        t.join()


if __name__ == '__main__':
    vf = VideoFeature(method='orb')
    vf.run(v_path='./video/test_countryroad_reverse.mp4')
