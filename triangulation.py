import cv2
import numpy as np
from featureDetect import FeatureDetect
from pose_estimation_2d2d import PoseEstimation2d2d


class Triangulate(object):
    def __init__(self, kds, matches, K, R, t):
        self.kds = kds
        self.matches = matches
        self.K = K
        self.R = R
        self.t = t

    def pixel2cam(self, point):
        u, v = point
        # print(self.K.I)
        # print(np.mat([u, v, 1]))
        cam_ = np.dot(self.K.I, np.mat([u, v, 1]).T)  # 归一化相机坐标 （x/z, y/z, 1）
        # print(np.array(cam_.T.tolist()[0][:2]))
        return cam_.T

    def triangulation(self, T1, T2, pts1, pts2):
        return cv2.triangulatePoints(T1, T2, pts1, pts2)

    def verify(self, points):
        # 验证重投影
        for i, matche in enumerate(self.matches):
            pt1_cam = self.pixel2cam(self.kds[0][0][matche.queryIdx].pt).tolist()[0][:2]
            print('The pt1_cam is :  ', pt1_cam)
            pt1_cam_3d = [points[i][0] / points[i][2], points[i][1] / points[i][2]]
            print('The pt1_cam_3d is :  ', pt1_cam_3d)
            print('d is : ', points[i][2])

            pt2_cam = self.pixel2cam(self.kds[1][0][matche.trainIdx].pt).tolist()[0]
            pt2_trans = np.array((self.R * np.mat([points[i][0], points[i][1], points[i][2]]).T + self.t).T.tolist()[0])
            # print(pt2_trans)
            pt2_trans /= pt2_trans[2]
            print('pt2_cam: ', pt2_cam)
            print('pt2_trans: ', pt2_trans)

    def run(self):
        T1 = np.mat([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]])
        T2 = np.mat([[self.R[0, 0], self.R[0, 1], self.R[0, 2], self.t[0, 0]],
                     [self.R[1, 0], self.R[1, 1], self.R[1, 2], self.t[1, 0]],
                     [self.R[2, 0], self.R[2, 1], self.R[2, 2], self.t[2, 0]]])
        # 将像素坐标转换到相机坐标
        pts1, pts2 = [], []
        for matche in self.matches:
            pts1.append(self.pixel2cam(self.kds[0][0][matche.queryIdx].pt).tolist()[0][:2])
            pts2.append(self.pixel2cam(self.kds[1][0][matche.trainIdx].pt).tolist()[0][:2])

        pts1, pts2 = np.array(pts1).T, np.array(pts2).T
        # print(pts1.shape)  # (2, 116)
        # 三角化
        tri_3d_4d = self.triangulation(T1, T2, pts1, pts2)
        # print(tri_3d_4d.shape[1])  # (4, 116)
        # 转换成非齐次坐标
        points = []
        for i in range(tri_3d_4d.shape[1]):
            x = tri_3d_4d[:, i]
            points.append((x[:3] / x[3]).tolist())
            # print(x)
            # print(x[3], type(x[3]))
            # print(x[:3] / x[3])
        # print(np.array(points))
        return np.array(points)


if __name__ == '__main__':
    # 相机内参
    K = np.mat([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
    images = ['./pics/1.png', './pics/2.png']

    fd = FeatureDetect('orb', draw=False)
    kds, matches = fd.run(images)

    pe = PoseEstimation2d2d(K, kds, matches)
    R, t = pe.run()

    tg = Triangulate(kds, matches, K, R, t)
    points = tg.run()
    # print(points)

    # 验证重投影
    # tg.verify(points)
