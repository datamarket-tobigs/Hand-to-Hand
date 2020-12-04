import cv2
import os

class video_to_img():
    def __init__(self):
        for i in range(20):
            os.makedirs('img/' + str(i * 10) + "-" + str((i + 1) * 10))
        self.video_path = "img/"
        self.video_dir = "video/"

    def video_to_frames(self, video, path_output_dir, i):
        # extract frames from a video and save to directory as 'x.png' where
        # x is the frame index
        vidcap = cv2.VideoCapture(video)
        count = 0
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(path_output_dir, '%d video %d.png') % (i, count), image)

                # 만약 이미지 너무 많아서 줄이고 싶으면 밑에 조건문 사용.
                # if count % 30 == 0:
                #     cv2.imwrite(os.path.join(path_output_dir, '%d video %d.png') % (i, count), image)
                count += 1

            else:
                break
        cv2.destroyAllWindows()
        vidcap.release()

    # 각 비디오 영상길이 평균좀 알아야할듯.
    def make_img(self):
        for i in range(20):
            now_video_path = self.video_path + str(i * 10) + "-" + str((i + 1) * 10)
            for j in range(10):
                video_name = self.video_dir + str(i * 10) + "/" + str(i * 10) + " " + str(j)
                self.video_to_frames(video_name + '.mp4', now_video_path, j)
            print("finish" + str(i * 10) + "folder")

if __name__ == "__main__":
    model = video_to_img()
    model.make_img()