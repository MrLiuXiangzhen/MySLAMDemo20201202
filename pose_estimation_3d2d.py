"""
    已知N个3D点以及与之匹配的2D点，求相机外参
"""
import numpy as np
import cv2
from featureDetect import FeatureDetect
from triangulation import Triangulate


class PoseEstimation3d2d(object):
    def __init__(self, pics, pics_depth, K, error=False):
        self.pics = pics
        self.pics_depth = pics_depth
        self.K = K
        self.error = error

    def get_feature(self):
        fd = FeatureDetect()
        return fd.run(self.pics)

    # 像素坐标转换相机坐标
    def pixel2cam(self, point, d=1):
        u, v = point[0] * d, point[1] * d
        # print(self.K.I)
        # print(np.mat([u, v, 1]))
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        world = np.mat([[x], [y], [d]])
        # print(world)
        # world = np.dot(self.K.I, np.mat([u, v, 1]).T)  # 归一化相机坐标 （x/z, y/z, 1）
        return world

    def pnp(self, pts_3d, pts_2d):
        retval, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, self.K, False, flags=cv2.SOLVEPNP_EPNP)
        # print(retval)
        # print(rvec)
        # print(tvec.T.tolist()[0])
        dst, jacobian = cv2.Rodrigues(rvec)
        # print(dst)
        return dst, tvec

    def run(self, l):
        # 获取特征点
        self.kds, self.matches = self.get_feature()
        d1 = cv2.imread(self.pics_depth[0], flags=0)
        # print(d1.shape)  # (480, 640)
        pts_3d, pts_2d = [], []
        # 3D 2D 匹配
        for matche in self.matches:
            # 深度提取
            d = d1[int(self.kds[0][0][matche.queryIdx].pt[1]), int(self.kds[0][0][matche.queryIdx].pt[0])]
            if not d: continue
            # print(d)
            w_pt1 = self.pixel2cam(self.kds[0][0][matche.queryIdx].pt, d)
            # print(np.array(w_pt1.T.tolist()[0]))
            pts_3d.append(np.array(w_pt1.T.tolist()[0]))  # 上一时刻的3d点
            pts_2d.append(list(self.kds[1][0][matche.trainIdx].pt))  # 下一时刻的2d点
            # print(pts_2d)
        # print('3D - 2D 匹配点数： ', len(pts_3d))
        R, t = self.pnp(np.mat(pts_3d), np.mat(pts_2d))  # 返回值R, t

        # 计算重投影误差
        if self.error:
            d2 = cv2.imread(self.pics_depth[1], flags=0)
            total_error = 0
            for matche in self.matches:
                d = d1[int(self.kds[0][0][matche.queryIdx].pt[1]), int(self.kds[0][0][matche.queryIdx].pt[0])]
                if not d: continue
                w_pt1 = self.pixel2cam(self.kds[0][0][matche.queryIdx].pt, d)
                point1 = np.array(self.kds[1][0][matche.trainIdx].pt)  # 65.55640517622955
                # print(pts_3d)
                # 计算三维点投影到二维平面上的坐标
                imagePoint, jacobian = cv2.projectPoints(w_pt1, np.mat(R), np.mat(t), self.K, 0)
                # print(imagePoint)  # [[[144.99988839 274.30642316]]]
                # print(imagePoint.tolist()[0][0])
                # print(point1)
                error = cv2.norm(np.array(point1.tolist()), np.array(imagePoint.tolist()[0][0]), cv2.NORM_L2) / len(self.matches)
                total_error += error
            print('The total_error is: ', total_error)
            # for matche in self.matches:
            #     point1 = self.kds[0][0][matche.queryIdx].pt
            #     point2 = self.kds[1][0][matche.trainIdx].pt
            #     d_1 = d1[int(self.kds[0][0][matche.queryIdx].pt[1]), int(self.kds[0][0][matche.queryIdx].pt[0])]
            #     d_2 = d2[int(self.kds[1][0][matche.trainIdx].pt[1]), int(self.kds[1][0][matche.trainIdx].pt[0])]
            #     # print(c_pt1 * d)
            #     if not d_1 or not d_2: continue
            #     w_pt1 = self.pixel2cam(point1, d_1)
            #     w_pt2 = self.pixel2cam(point2, d_2)
            #
            #     pw_pt2 = (R * w_pt1) + t
            #     # print(w_pt2)
            #     # print(pw_pt2)
            #     error = w_pt2 - pw_pt2
            #     print(error)

        return R, t


if __name__ == '__main__':
    # 相机内参
    K = np.mat([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    l = np.mat([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 2]])

    pe = PoseEstimation3d2d(['./pics/1.png', './pics/2.png'], ['./pics/1_depth.png', './pics/2_depth.png'], K, error=False)
    R, t = pe.run(l)
    print(R)
    print(t)
'''
[[-0.9825239   0.04961937 -0.17940093]
 [ 0.02305065 -0.92396013 -0.3817936 ]
 [-0.18470367 -0.37925665  0.90666915]]
[[  8.56503748]
 [ 10.56259756]
 [-70.25761862]]

Process finished with exit code 0
'''