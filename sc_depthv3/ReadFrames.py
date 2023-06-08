
import cv2
import os
import glob
from datetime import datetime
import numpy as np
from PIL import Image

def frames2video(videoname):
    cap = cv2.VideoCapture("D:/ILSVRC/Data/VID/snippets/test/"
                            + videoname)
    im_list = os.listdir('../data/demo/img/')
    im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  # 最好再看看图片顺序对不
    img = Image.open(os.path.join('../data/demo/img/', im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter('../output/demo/'+ videoname, fourcc, 1, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join('../data/demo/img/' + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('finish')



if __name__ == '__main__':
    # cap = cv2.VideoCapture("E:\\fyx\\research\\professor\\恶劣检测\\DATA0\\ILSVRC2015\\ILSVRC20150\\Data\\VID\\snippets\\train\\ILSVRC2015_VID_train_0000\\"
    #                        "ILSVRC2015_train_00018003.mp4")
    # im_dir = '../data/demo/img/'  # 合成视频存放的路径
    # video_dir = '../output/demo/'  # 帧存放路径
    # fps = 10  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    frames2video('ILSVRC2015_test_00000000.mp4')