
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data
from tqdm import trange

from aputils import Video, camera_config
from scripts._path_init import *
from visualDet3D.data.pipeline import build_augmentator


class NuScenesMonoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        video_dir: str,
        scene: str = "scene-0757-CAM_FRONT"
    ):
        self.output_dict = {
                "calib": False,
                "image": True,
                "label": False,
                "velodyne": False
            }
        self.transform = build_augmentator(cfg.data.test_augmentation)
        self.obj_types = cfg.obj_types

        with open(os.path.join(video_dir, "frames.pickle"), "rb") as f:
            video = pickle.load(f)[scene]
        
        videofile = video['filename']
        frames = [camera_config(*f, 0) for f in video['frames']]
        start = video['start']
        self.videofile = os.path.join(video_dir, videofile)
        self.video = Video(self.videofile, frames, start)
        self._images = []
        cap = cv2.VideoCapture(os.path.join(video_dir, videofile))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # preload all the images
        for _ in trange(self.frame_count):
            ret, frame = cap.read()
            assert ret
            self._images.append(frame)
        assert not cap.read()[0]
        cap.release()
        cv2.destroyAllWindows()
        
        self._P2s = [np.concatenate([
            np.array(f.camera_intrinsic),
            np.array([[0, 0, 0]]).T,
        ], axis=1) for f in self.video]

    
    def __len__(self):
        return len(self.video)

    def __getitem__(self, index: "int"):
        image = self._images[index]
        p2 = self._P2s[index]
        frame = self.video[index]

        t_image, t_p2 = self.transform(image, p2=deepcopy(p2))
        output_dict = {
            'calib': t_p2,
            'image': t_image,
            'original_shape':image.shape,
            'original_P':p2.copy(),
            'original_image': image,
            'timestamp': frame.timestamp,
            'camera_rotation': frame.camera_rotation,
            'camera_translation': frame.camera_translation,
            'camera_intrinsic': frame.camera_intrinsic,
        }
        return output_dict

    @staticmethod
    def collate_fn(batch):
        rgb_images = np.array([
            item["image"]
            for item in batch
        ])  # [batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        calib = [item["calib"] for item in batch]
        return torch.from_numpy(rgb_images).float(), calib 


@dataclass
class Detection:
    bbox_2d: "List[Any]"
    clss: "List[int]"
    bbox_3d_state_3d: "List[Any]"
    thetas: "List[Any] | npt.NDArray[np.floating[Any]]"
    scores: "List[Any]"


def format_detections(
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
        bbox_3d_state_3d[i][1] = bbox_3d_state_3d[i][1] + 0.5 * bbox_3d_state_3d[i][4]  # kitti receive bottom center

    if thetas is None:
        thetas = np.ones(bbox_2d.shape[0]) * -10
    if len(scores) > 0:
        for i in range(len(bbox_2d)):
            if scores[i] < threshold:
                continue
        return Detection(
            bbox_2d=bbox_2d,
            clss=obj_types,
            bbox_3d_state_3d=bbox_3d_state_3d,
            thetas=thetas,
            scores=scores,
        )

    return Detection([], [], [], [], [])
