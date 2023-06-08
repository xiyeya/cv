
import cv2
import os
import glob
from datetime import datetime
import numpy as np
from PIL import Image

def video_to_frame(cap, in_dir):
    # cap = cv2.VideoCapture("./query_video/test_video_0.mp4")
    c = 1
    frameRate = 1  # 帧数截取间隔（每隔100帧截取一帧）

    while (True):
        ret, frame = cap.read()
        if ret:
            if (c % frameRate == 0):
                print("开始截取视频第：" + str(c) + " 帧")
                # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                cv2.imwrite(in_dir + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
            c += 1
        else:
            print("所有帧都已经保存完成")
            break
    cap.release()


def frames2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  # 最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('finish')



if __name__ == '__main__':
    cap = cv2.VideoCapture("D:/ILSVRC/Data/VID/snippets/test/ILSVRC2015_test_00043015.mp4")
    im_dir = './demo/input/'  # 合成视频存放的路径
    video_dir = './vid/1.avi'  # 帧存放路径
    fps = 10  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅

    video_to_frame(cap, im_dir)
    # frames2video(im_dir, video_dir, fps)