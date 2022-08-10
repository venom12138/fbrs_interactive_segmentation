from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

from collections import namedtuple
Click = namedtuple('Click', ['is_positive', 'coords'])

def evaluate_dataset(dataset, predictor, oracle_eval=False, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        item = dataset[index]
        # 没有用到
        if oracle_eval:
            gt_mask = torch.tensor(sample['instances_mask'], dtype=torch.float32)
            
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            predictor.opt_functor.mask_loss.set_gt_mask(gt_mask)
        _, sample_ious, _ = evaluate_sample(item['images'], sample['instances_mask'], predictor, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

def evaluate_sample(image_nd, instances_mask, predictor, max_iou_thr,
                    pred_thr=0.49, max_clicks=20, pos_clicks=None, neg_clicks=None):
    if instances_mask is not None:
        assert pos_clicks is None and neg_clicks is None
        clicker = Clicker(gt_mask=instances_mask)
    else:
        assert (pos_clicks is not None) and (neg_clicks is not None)
        pos_clicks = [Click(is_positive=True, coords=(coords_y, coords_x)) for coords_x, coords_y in pos_clicks]
        neg_clicks = [Click(is_positive=False, coords=(coords_y, coords_x)) for coords_x, coords_y in neg_clicks]
        pos_clicks.extend(neg_clicks)
        clicks = pos_clicks
        print(clicks)
        clicker = None
    
    pred_mask = np.zeros_like(instances_mask)
    ious_list = []
    if clicker is None:
        with torch.no_grad():
            predictor.set_input_image(image_nd)
            pred_probs = predictor.get_prediction(clicks)
            pred_mask = pred_probs > pred_thr
        return clicks, None, pred_probs
    else:
        with torch.no_grad():
            predictor.set_input_image(image_nd)
            for click_number in range(max_clicks):
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker)
                pred_mask = pred_probs > pred_thr

                iou = utils.get_iou(instances_mask, pred_mask)
                ious_list.append(iou)

                if iou >= max_iou_thr:
                    break
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
