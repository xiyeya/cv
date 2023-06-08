
import cv2
import os
import glob
from datetime import datetime
import numpy as np
from PIL import Image

def video_to_frame(cap):
    # cap = cv2.VideoCapture("./query_video/test_video_0.mp4")
    c = 1
    frameRate = 10  # 帧数截取间隔（每隔100帧截取一帧）

    while (True):
        ret, frame = cap.read()
        if ret:
            if (c % frameRate == 0):
                print("开始截取视频第：" + str(c) + " 帧")
                # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                cv2.imwrite("./frames/" + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
            c += 1
            cv2.waitKey(0)
        else:
            print("所有帧都已经保存完成")
            break
    cap.release()


if __name__ == '__main__':
    cap = cv2.VideoCapture("E:\\fyx\\research\\professor\\恶劣检测\\DATA0\\ILSVRC2015\\ILSVRC20150\\Data\\VID\\snippets\\train\\ILSVRC2015_VID_train_0000\\"
                           "ILSVRC2015_train_00018003.mp4")
    im_dir = '../data/demo/img/'  # 合成视频存放的路径
    fps = 10  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    frames2video(im_dir, video_dir, fps)