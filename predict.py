import os
from tqdm import tqdm
from easydict import EasyDict
from typing import List, Sized
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data
import torch.nn as nn
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from visualDet3D.networks.utils.utils import BBox3dProjector, BackProjection


@PIPELINE_DICT.register_module
@torch.no_grad()
def predict(
    cfg:EasyDict, 
    model:nn.Module,
    dataset_val:Sized,
):
    model.eval()
    result_path = os.path.join(cfg.path.preprocessed_path, 'data')
    if os.path.isdir(result_path):
        os.system("rm -r {}".format(result_path))
        print("clean up the recorder directory of {}".format(result_path))
    os.mkdir(result_path)
    print("rebuild {}".format(result_path))
    test_func = PIPELINE_DICT[cfg.trainer.test_func]
    projector = BBox3dProjector().cuda()
    backprojector = BackProjection().cuda()
    detections = []
    for index in tqdm(range(len(dataset_val))):
        detections.append(test_one(cfg, index, dataset_val, model, test_func, backprojector, projector))
    return detections


def test_one(
    cfg: "EasyDict",
    index: int,
    dataset: "torch.utils.data.Dataset",
    model: "nn.Module",
    test_func,
    backprojector: "BackProjection",
    projector: "BBox3dProjector"
):
    data = dataset[index]
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']
    collated_data = dataset.collate_fn([data])
        
    scores, bbox, obj_names = test_func(collated_data, model, None, cfg=cfg)
    bbox_2d = bbox[:, 0:4]
    if bbox.shape[1] <= 4:
        raise Exception('Should run 3D')
    bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha, bot, top]
    bbox_3d_state_3d = backprojector(bbox_3d_state, P2) #[x, y, z, w,h ,l, alpha, bot, top]

    _, _, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

    original_P = data['original_P']
    scale_x = original_P[0, 0] / P2[0, 0]
    scale_y = original_P[1, 1] / P2[1, 1]
    
    shift_left = original_P[0, 2] / scale_x - P2[0, 2]
    shift_top  = original_P[1, 2] / scale_y - P2[1, 2]
    bbox_2d[:, 0:4:2] += shift_left
    bbox_2d[:, 1:4:2] += shift_top

    bbox_2d[:, 0:4:2] *= scale_x
    bbox_2d[:, 1:4:2] *= scale_y

    return write_result_to_file(scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)


def write_result_to_file(
    scores,
    bbox_2d,
    bbox_3d_state_3d=None,
    thetas=None,
    obj_types=['Car', 'Pedestrian', 'Cyclist'],
    threshold=0.4
):
    """Output Kitti prediction results of one frame

    Args:
        base_result_path (str): path to the result dictionary 
        index (int): index of the target frame
        scores (List[float]): A list or numpy array or cpu tensor of float for score
        bbox_2d (np.ndarray): numpy array of [N, 4]
        bbox_3d_state_3d (np.ndarray, optional): 3D stats [N, 7] [x_center, y_center, z_center, w, h, l, alpha]. Defaults to None.
        thetas (np.ndarray, optional): [N]. Defaults to None.
        obj_types (List[str], optional): List of string if object type names. Defaults to ['Car', 'Pedestrian', 'Cyclist'].
        threshold (float, optional): Threshold for selection samples. Defaults to 0.4.
    """    
    if bbox_3d_state_3d is None:
        raise Exception('should run 3d')

    for i in range(len(bbox_2d)):
        bbox_3d_state_3d[i][1] = bbox_3d_state_3d[i][1] + 0.5*bbox_3d_state_3d[i][4] # kitti receive bottom center

    if thetas is None:
        thetas = np.ones(bbox_2d.shape[0]) * -10
    if len(scores) > 0:
        for i in range(len(bbox_2d)):
            if scores[i] < threshold:
                continue
        return {
            "bbox_2d": bbox_2d,
            "obj_types": obj_types,
            "bbox_3d_state_3d": bbox_3d_state_3d,
            "thetas": thetas,
            "scores": scores,
        }
    return None
