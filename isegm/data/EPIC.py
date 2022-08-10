from pathlib import Path

import cv2
import numpy as np
import glob
from .base import ISDataset
import yaml
import os

class EPIC_KitchenDataset(ISDataset):
    def __init__(self, dataset_path,
                images_dir_name='first_last_frame', masks_dir_name=None,
                 **kwargs):
        super(EPIC_KitchenDataset, self).__init__(**kwargs)
        # dataset_path = ./datasets/EPIC-Kitchen/P04
        self.dataset_path = Path(dataset_path)
        # /cluster/home2/yjw/venom/fbrs_interactive_segmentation/datasets/EPIC-Kitchen/P04/first_last_frame
        self._images_path = self.dataset_path / images_dir_name
        if masks_dir_name is not None:
            # TODO: 待测试
            self._insts_path = self.dataset_path / masks_dir_name
            self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        else:
            self._insts_path = None
            self._masks_paths = None
        # [datasets/EPIC-Kitchen/P04/first_last_frame/P04_01/8157/frame_000001.jpg, ...]
        self.dataset_samples = glob.glob(f'{self._images_path}/*/*/*.jpg')
        with open(self.dataset_path / 'first_last_frame.yaml') as f:
            self.clicks_info = yaml.safe_load(f)

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = image_name
        # TODO: 暂无mask path
        # mask_path = str(self._masks_paths[image_name.split('/')split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        # instances_mask[instances_mask == 128] = -1
        # instances_mask[instances_mask > 128] = 1
        instances_mask = None
        instances_ids = [1]

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }
        img_info = {'uid':image_path.split('/')[-2],'video_id':image_path.split('/')[-3],
            'img_name':image_path.split('/')[-1]}
        try:
            neg_clicks = self.clicks_info[img_info['uid']][str(os.path.join(img_info['video_id'], img_info['uid'], img_info['img_name']))]['neg_click']
            pos_clicks = self.clicks_info[img_info['uid']][str(os.path.join(img_info['video_id'], img_info['uid'], img_info['img_name']))]['pos_click']
        except:
            neg_clicks = []
            pos_clicks = []
        clean_neg_clicks = []
        clean_pos_clicks = []
        for x,y in neg_clicks:
            if x > 30 and y >30:
                clean_neg_clicks.append((x,y))
        for x,y in pos_clicks:
            if x > 30 and y > 30:
                clean_pos_clicks.append((x,y))
        # instances_mask不使用
        return {
            'image': image,
            'instances_mask': image,
            'instances_info': instances_info,
            'image_id': index,
            'img_info':img_info,
            'neg_clicks':clean_neg_clicks,
            'pos_clicks':clean_pos_clicks
        }
if __name__ == '__main__':
    EPIC_KitchenDataset('/cluster/home2/yjw/venom/fbrs_interactive_segmentation/datasets/EPIC-Kitchen/P04')