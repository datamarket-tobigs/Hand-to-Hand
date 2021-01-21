import numpy as np
import cv2
import json
import os
from helpers import make_dir


def create_stick(filename, keypoints, save_path):
    """
    Make a stick image given a json keypoints file
    
    :param string filename: name of image to save
    :param array keypoints
    :param string save_path: path to save image
    """
    
    # Create keypoint pairs
    pose_point_pair = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 9], [9, 8], [8, 10], [10, 5]]

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
    
    
    for keypoint in range(len(keypoints)):
        pose = keypoints[keypoint][:22]
        face = keypoints[keypoint][30:170]
        left_hand = keypoints[keypoint][170:212]
        right_hand = keypoints[keypoint][212:255]

        part = [pose, face, left_hand, right_hand]
        part_num_points = [11, 68, 21, 21]
        part_pair = [pose_point_pair, face_point_pair, hand_point_pair, hand_point_pair]

        # Create paper
        img = np.zeros((1500, 1500), np.uint8)+255

        for p in range(len(part)):
            x = part[p][0::2]
            y = part[p][1::2]

            # Draw points
            for i in range(part_num_points[p]):
                cv2.circle(img, (int(x[i]*2048), int(y[i]*1152)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)  
            
            # Draw lines
            for pair in part_pair[p]:
                cv2.line(img, (int(x[pair[0]]*2048), int(y[pair[0]]*1152)), (int(x[pair[1]]*2048), int(y[pair[1]]*1152)), (0, 0, 255), 2)
           
        # Write the image frame
        cv2.imwrite(save_path + f'/{filename[:-5]}_{keypoint:03}.jpg', img)
    
    return

################################################

def create_video(save_path):
    """
    Make a video given a image

    :param string save_path: path to save video
    """
    
    # Load stick imgages
    images = [img for img in os.listdir(save_path)]
    images.sort()

    # Image to video
    fps = 30

    frame_array = []
    for i in range(len(images)):
        filename = save_path + images[i]

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        frame_array.append(img)

    # Make Video
    out = cv2.VideoWriter(save_path + str(images[0])[:-8] + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    # Write the video frame
    for i in range(len(frame_array)):
        out.write(frame_array[i])

    # Release the video
    out.release()
    
    return

################################################

def create_img_video(file_path, save_path, filename): 
    """
    Load json file and Make a stick image and video

    :param string file_path: path to load json file
    :param string save_path: path to save image and video
    :param string filename: name of image and video to save
    """
    
    # Open keypoints json file in file_path
    f = open(file_path + filename, encoding="UTF-8")
    keypoints = json.loads(f.read())

    save_path = save_path + filename[:-5] + '/'
    # Call func make_dir
    make_dir(save_path)
    
    # Make images
    create_stick(filename, keypoints, save_path)
    # Make video
    create_video(save_path)
            
    print(filename.split(".json")[0], '  img, video saving Complete!')
    
    return