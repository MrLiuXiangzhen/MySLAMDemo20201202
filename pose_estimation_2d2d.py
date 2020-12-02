"""
    已知N个匹配的2d点, 求相机外参
"""
import cv2
import numpy as np
from featureDetect import FeatureDetect


class PoseEstimation2d2d(object):
    def __init__(self, K, kds, matches):
        '''
        :param K: 相机内参矩阵
        :param kds: 关键点，描述子
        :param matches: 匹配关系
        '''
        self.K = K
        self.kds = kds
        self.matches = matches

    # 基础矩阵
    @staticmethod
    def fundamental(points1, points2):
        return cv2.findFundamentalMat(points1, points2)

    # 本质矩阵
    @staticmethod
    def principal(points1, points2, principal_point, focal_length):
        '''
        :param points1: 图1 匹配点坐标
        :param points2: 图2 匹配点坐标
        :param principal_point: 光心，类型：元组
        :param focal_length: 焦距
        :return: 本质矩阵 [E, mask]
        '''
        return cv2.findEssentialMat(points1, points2, principal_point, focal_length)

    # 单应矩阵
    @staticmethod
    def homography(points1, points2):
        return cv2.findHomography(points1, points2)

    # 由本质矩阵恢复外参
    def rec_pose(self, E, points1, points2):
        return cv2.recoverPose(E, points1, points2, cameraMatrix=self.K)

    # 像素坐标转换相机坐标
    def pixel2cam(self, point):
        u, v = point
        # print(self.K.I)
        # print(np.mat([u, v, 1]))
        cam_ = np.dot(self.K.I, np.mat([u, v, 1]).T)  # 归一化相机坐标 （x/z, y/z, 1）
        return cam_

    # 验证
    def verify(self, R, t):
        '''
        :param R: 旋转矩阵
        :param t: 平移参数
        :return:  print(v_E, d)
        '''
        # E = 反对称(t) * R * scale
        t_ = np.mat([[0, -t[2][0], t[1][0]], [t[2][0], 0, -t[0][0]], [-t[1][0], t[0][0], 0]])
        R = np.mat(R)
        v_E = t_ * R
        print("v_E:   ", v_E)
        # 验证对极约束
        for matche in self.matches:
            # 像素坐标到相机坐标
            pt1 = self.pixel2cam(self.kds[0][0][matche.queryIdx].pt)
            pt2 = self.pixel2cam(self.kds[1][0][matche.trainIdx].pt)
            d = pt2.T * v_E * pt1
            print('d:  ', d)
        # return v_E, d

    def run(self):
        '''
        :return:  R, t
        '''
        points1, points2 = [], []
        # keypoints1, keypoints2 = kds[0][0], kds[1][0]
        # print(len(kds[0][0]))
        for i in range(len(self.matches)):
            # print(type(kds[0][0][matches[i].queryIdx].pt))  # <class 'tuple'>
            points1.append(list(self.kds[0][0][self.matches[i].queryIdx].pt))  # keypoints1 -> self.kds[0][0]
            points2.append(list(self.kds[1][0][self.matches[i].trainIdx].pt))  # keypoints2 -> self.kds[1][0]
        # print(len(point1))
        # print(len(point2))
        points1, points2 = np.array(points1), np.array(points2)

        # 计算基础矩阵
        # F = self.fundamental(points1, points2)
        # print(fundamental_matrix)

        # 计算本质矩阵
        focal_length = 521  # 焦距
        principal_point = (325.1, 249.7)  # 光心
        E, mask = self.principal(points1, points2, focal_length, principal_point)
        # print("E:    ", E)

        # 计算单应矩阵
        # H = self.homography(points1, points2)

        # 从本质矩阵恢复平移旋转信息
        retval, R, t, mask = self.rec_pose(E, points1, points2)

        return R, t


if __name__ == '__main__':
    # 相机内参
    K = np.mat([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
    images = ['./pics/1.png', './pics/2.png']

    # 特征检测与匹配
    fd = FeatureDetect('orb', draw=False)
    kds, matches = fd.run(images)
    # print(kds)

    # 外参计算
    pe = PoseEstimation2d2d(K, kds, matches)
    R, t = pe.run()
    print(R)
    print(t)

    # 验证
    # pe.verify(R, t)
'''
[[ 0.99513998 -0.05834222  0.07932597]
 [ 0.05466064  0.99735936  0.04781754]
 [-0.08190628 -0.04324914  0.9957012 ]]
[[-0.97942344]
 [-0.19966383]
 [ 0.02939523]]

Process finished with exit code 0
'''