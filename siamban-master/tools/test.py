# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.bbox import get_axis_aligned_bbox
from siamban.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
# from toolkit.utils.region import vot_overlap, vot_float2str


config_path = os.path.join(os.path.abspath('..'),'experiments/siamban_r50_l234_otb/config.yaml')
snapshot_path = os.path.join(os.path.abspath('..'),'experiments/siamban_r50_l234_otb/model.pth')

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', default='OTB50', type=str,
        help='datasets')
parser.add_argument('--config', default=config_path, type=str,
        help='config file')


parser.add_argument('--snapshot', default=snapshot_path, type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true', default=True,
        help='whether visualzie result')
parser.add_argument('--gpu_id', default='not_set', type=str, 
        help="gpu id")

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config) # 加载配置文件

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, 'testing_dataset', args.dataset)

    # create model
    model = ModelBuilder() # 创建SiamBAN模型架构

    # load model
    model = load_pretrain(model, args.snapshot).eval() # 加载预训练好的模型

    # build tracker
    tracker = build_tracker(model) # 构建追踪器架构

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset, # 创建加载数据集，会进入otb.py文件
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount() # 用于统计推理时间
            if idx == 0: # 如果是第一帧进行如下操作：
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox)) # 获取OTB50.json文件中对第一帧的目标框
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] # 获取第一帧的目标框
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency()) # 计算本帧的推理时间

            # 追踪结果可视化语句
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results 保存追踪结果
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
    print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
