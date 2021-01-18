import os
import cv2

if __name__ == '__main__':
    path = './dataset/WORD_REAL003/stick/NIA_SL_WORD0829_REAL02_F/'

    images = [img for img in os.listdir(path)]
    images.sort()

    fps = 30

    frame_array = []
    for i in range(len(images)):
        filename = path + images[i]
        
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(path+'test.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    
    out.release()
