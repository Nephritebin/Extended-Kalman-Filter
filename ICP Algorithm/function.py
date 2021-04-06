import numpy as np
from plyfile import *


class Frame:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.robot_pos = np.mat([0, 1, 0])

    def move(self, rotate, translate):
        self.rotate = rotate
        self.translate = translate


# Read the data from route
def readfile(route):
    data = PlyData.read(route)

    x = data['vertex']['x']
    y = data['vertex']['y']
    z = data['vertex']['z']

    frame = Frame(x, y, z)
    return frame


# Match the points in different sets using violent search
# x0, y0 are the initial positions and x1, y1 are the new positions, and datatype is np.mat, shape is (1, n)
def match(x1, y1, x2, y2):
    new_x1 = []
    new_y1 = []
    new_x2 = []
    new_y2 = []
    distance = []
    # print(x2.shape[1])

    for i in range(x2.shape[1]):
        dis = np.power(x1 - x2[0, i] * np.ones(x1.shape), 2) + np.power(y1 - y2[0, i] * np.ones(x1.shape), 2)
        index = np.argmin(dis)
        # print(x2[0, index])
        new_x2.append(x2[0, i])
        new_y2.append(y2[0, i])
        new_x1.append(x1[0, index])
        new_y1.append(y1[0, index])
        distance.append(dis[0, index])

    new_x2 = np.mat(new_x2)
    new_y2 = np.mat(new_y2)
    new_x1 = np.mat(new_x1)
    new_y1 = np.mat(new_y1)
    distance = np.mat(distance)

    pos = np.r_[new_x1, new_y1]
    pos_new = np.r_[new_x2, new_y2]

    return pos, pos_new, distance


# Eliminate the error values bigger than threshold
# pos0 is the previous position and pos1 is the new position, and datatype is np.mat, shape is (2, n)
def under_threshold(pos0, pos1, distance, threshold):
    two_frame_pos = np.r_[pos0, pos1, distance]
    # print(two_frame_pos.shape)

    index = np.where(two_frame_pos[4, :] >= threshold)[1]
    new_pos = np.delete(two_frame_pos, index, axis=1)
    # print(new_pos.shape)

    return new_pos[0:2, :], new_pos[2:4, :]


# Calculate the R and t using SVD, pos0 = R * pos1 + t
# pos1 is the previous position and pos2 is the new position, and datatype is np.mat, shape is (2, n)
def svd_optimize(pos0, pos1):
    pos0_no_center = pos0 - pos0.sum(axis=1) / pos0.shape[1]
    pos1_no_center = pos1 - pos1.sum(axis=1) / pos1.shape[1]

    w = np.matmul(pos1_no_center, np.transpose(pos0_no_center))
    u, s, vh = np.linalg.svd(w)
    rotate = np.matmul(np.transpose(vh), np.transpose(u))
    translate = (pos0.sum(axis=1) / pos0.shape[1]) - np.matmul(rotate, pos1.sum(axis=1) / pos1.shape[1])

    return rotate, translate


# Calculate the final result of transform matrix R and t from several times iteration
# x0, y0 are the initial positions and x1, y1 are the new positions, and datatype is np.mat, shape is (1, n)
# rotate is the matrix R and translate is the vector t
def transform(x0, y0, x1, y1, threshold=10, iteration=200):
    rotate = np.diag([1, 1])
    translate = np.transpose(np.mat([0, 0]))
    vector_homo = np.mat([0, 0, 1])

    trans_mat = np.r_[np.c_[rotate, translate], vector_homo]

    flag = 0
    while True:
        flag = flag + 1
        pos0, pos1, distance = match(x0, y0, x1, y1)
        pos0, pos1 = under_threshold(pos0, pos1, distance, threshold)
        rotate, translate = svd_optimize(pos0, pos1)

        if np.linalg.det(rotate) < 0:
            rotate = -rotate

        trans_mat_new = np.r_[np.c_[rotate, translate], vector_homo]
        trans_mat = np.matmul(trans_mat_new, trans_mat)

        pos_new = np.matmul(rotate, np.r_[x1, y1]) + translate

        x1 = pos_new[0, :]
        y1 = pos_new[1, :]

        if flag >= iteration:
            break

        # pos0_center = [sum(x0.tolist()[0]) / len(x0.tolist()[0]), sum(y0.tolist()[0]) / len(y0.tolist()[0])]
        # pos_new_center = [sum(x1.tolist()[0]) / len(x1.tolist()[0]), sum(y1.tolist()[0]) / len(y1.tolist()[0])]
        # center_distance = (pos_new_center[0] - pos0_center[0]) ** 2 + (pos_new_center[1] - pos0_center[1]) ** 2
        # center_distance = np.sqrt(center_distance)
        #
        # print(center_distance)
        # if center_distance <= 0.1: # limitation:
        #     break

    rotate = trans_mat[0:2, 0:2]
    translate = trans_mat[0:2, 2]
    print(trans_mat)

    return rotate, translate, pos_new
