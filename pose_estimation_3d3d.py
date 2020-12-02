"""
    已知N个匹配的3D点，求相机外参
"""
import numpy as np
import cv2
from featureDetect import FeatureDetect


class PoseEstimation3d3d(object):
    def __init__(self, K):
        self.K = K

    # 像素坐标转换世界坐标
    def pixel2cam(self, point, d=1):
        u, v = point[0] * d, point[1] * d
        # print(self.K.I)
        # print(np.mat([u, v, 1]))
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        # world = (self.K * np.mat([u, v, 1]).T) * d
        # world = np.dot(self.K, np.mat([u, v, d]).T)  # 归一化相机坐标 （x, y, z）
        world = np.mat([[x], [y], [d]])
        # print(self.K)
        # print(np.mat([u, v, d]).T)
        # print(world)
        return world

    def run(self, pics, pics_depth):
        fd = FeatureDetect('orb')
        kds, matches = fd.run(pics)
        d1 = cv2.imread(pics_depth[0], flags=0)
        d2 = cv2.imread(pics_depth[1], flags=0)
        w_pts1, w_pts2 = [], []
        sum_pt1, sum_pt2 = np.mat([[0.0], [0.0], [0.0]]), np.mat([[0.0], [0.0], [0.0]])
        for matche in matches:
            d_1 = d1[int(kds[0][0][matche.queryIdx].pt[1]), int(kds[0][0][matche.queryIdx].pt[0])]
            d_2 = d2[int(kds[1][0][matche.trainIdx].pt[1]), int(kds[1][0][matche.trainIdx].pt[0])]
            if not d_1 or not d_2: continue
            w_pt1 = self.pixel2cam(kds[0][0][matche.queryIdx].pt, d_1)
            # print(c_pt1)
            w_pt2 = self.pixel2cam(kds[1][0][matche.trainIdx].pt, d_2)

            sum_pt1 += w_pt1
            sum_pt2 += w_pt2
            w_pts1.append(w_pt1)
            w_pts2.append(w_pt2)

        N = len(w_pts1)
        # print(N)
        e_pt1 = sum_pt1 / N
        e_pt2 = sum_pt2 / N
        # print(e_pt1)

        r_pts1, r_pts2 = [], []
        # 均值移除
        for i in range(N):
            r_pts1.append(w_pts1[i] - e_pt1)
            r_pts2.append(w_pts2[i] - e_pt2)
            # print(w_pts1[i])
            # print(w_pts1[i] - e_pt1)
        # SVD求解法
        W = np.mat([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for i in range(len(r_pts1)):
            W += r_pts1[i] * r_pts2[i].T
        # print(W)
        U, S, V = np.linalg.svd(W)
        # print(U)
        R = U * V.T
        # print(R)
        t = e_pt1 - R * e_pt2
        # print(t)
        return R, t


if __name__ == '__main__':
    # 相机内参
    K = np.mat([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    pe = PoseEstimation3d3d(K)
    R, t = pe.run(['./pics/1.png', './pics/2.png'], ['./pics/1_depth.png', './pics/2_depth.png'])
    print(R)
    print(t)
'''
[[-0.09385628 -0.60922846  0.78742091]
 [ 0.97204263  0.11490783  0.2047665 ]
 [-0.21523041  0.78462531  0.58141121]]
[[-66.61226288]
 [  2.92498554]
 [ 18.83358582]]

Process finished with exit code 0

[[ 0.71035176  0.48742494 -0.50775714]
 [-0.53533602  0.84251643  0.05984482]
 [ 0.45696359  0.22930982  0.85941915]]
[[15.94038196]
 [ 8.96427839]
 [-6.5191628 ]]

Process finished with exit code 0
'''