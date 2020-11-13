import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os

pose_point_pair_ver1 = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 9], [9, 8], [8, 12], [12, 1]]
pose_point_pair_ver2 = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 9], [9, 8], [8, 12], [12, 5]]

face_point_pair = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], 
    [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], 
    [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [25, 26], [27, 28], [28, 29], 
    [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [36, 37], [37, 38], 
    [38, 39], [39, 40], [40, 41], [36, 41], [42, 43], [43, 44], [44, 45], [45, 46], 
    [46, 47], [42, 47], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], 
    [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [48, 59], [60, 61], [61, 62], [62, 63], [63, 64]
    ]

hand_point_pair = [[i, i+1] for i in range(0, 20)]

def draw_keypoints(img, data, num_points, pair):
    x = data[0::3]
    y = data[1::3]

    for i in range(num_points):
        cv2.circle(img, (int(x[i]), int(y[i])), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)  

    for p in pair:
        cv2.line(img, (int(x[p[0]]), int(y[p[0]])), (int(x[p[1]]), int(y[p[1]])), (0, 0, 255), 2)



def create_stick(file, keypoints):
    # 흰 종이를 꺼냅니다.
    img = np.zeros((1500, 1500), np.uint8)+255

    part = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    part_num_points = [25, 68, 21, 21]
    part_pair = [pose_point_pair_ver2, face_point_pair, hand_point_pair, hand_point_pair]

    for i in range(len(part)):
        draw_keypoints(img, keypoints['people'][0][part[i]], part_num_points[i], part_pair[i])
    
    cv2.imwrite(f'./dataset/WORD_REAL003/stick/NIA_SL_WORD0829_REAL02_F/{file[:-5]}.jpg', img)


if __name__ == '__main__':
    path = './dataset/WORD_REAL003/Keypoint_Json/NIA_SL_WORD0829_REAL02_F.mp4/'

    for json_file in os.listdir(path):
        f = open(path + json_file, encoding="UTF-8")
        keypoints = json.loads(f.read())

        create_stick(json_file, keypoints)
    