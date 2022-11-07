import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List
import sys
from pathlib import Path
import platform

import cv2
import fire
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.utils.data
from easydict import EasyDict
from tqdm import tqdm

from aputils import Video, camera_config
from scripts._path_init import *
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.networks.detectors.monodtr_detector import MonoDTR
from visualDet3D.networks.pipelines.testers import test_mono_detection
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector
from visualDet3D.utils.utils import cfg_from_file

if "./submodules" not in sys.path:
    sys.path.append("./submodules")

FILE = Path(__file__).resolve()
SUBMODULE = FILE.parent / "submodules"
YOLOV5_STRONGSORT_OSNET_ROOT = SUBMODULE / "Yolov5_StrongSORT_OSNet"

if str(YOLOV5_STRONGSORT_OSNET_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT))
if str(YOLOV5_STRONGSORT_OSNET_ROOT / "strong_sort") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT / "strong_sort"))
if str(YOLOV5_STRONGSORT_OSNET_ROOT / "strong_sort/deep/reid") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT / "strong_sort/deep/reid"))

from submodules.Yolov5_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT
from submodules.Yolov5_StrongSORT_OSNet.strong_sort.utils.parser import get_config

print('CUDA available: {}'.format(torch.cuda.is_available()))

VIDEO_DIR = "./data/boston-seaport/"


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device(arg)


class NuScenesMonoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, video_dir: str, scene: str = "scene-0757-CAM_FRONT"):
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
        self.images = []
        cap = cv2.VideoCapture(os.path.join(video_dir, videofile))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.images.append(frame)
        assert len(self.images) == len(self.video), (len(self.images), len(self.video))
        cap.release()
        cv2.destroyAllWindows()
    
    def __len__(self):
        return len(self.video)

    def __getitem__(self, index):
        frame = self.video[index]
        image = self.images[index][:, :, ::-1].copy()
        p2 = np.concatenate([frame.camera_intrinsic, np.array([[0, 0, 0]]).T], axis=1)
        transformed_image, transformed_P2 = self.transform(image, p2=deepcopy(p2))

        output_dict = {'calib': transformed_P2,
                       'image': transformed_image,
                       'original_shape':image.shape,
                       'original_P':p2.copy()}
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
    obj_types: "List[str]"
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
            obj_types=obj_types,
            bbox_3d_state_3d=bbox_3d_state_3d,
            thetas=thetas,
            scores=scores,
        )
    return Detection([], [], [], [], [])


def main(
    config: str = "config/config.py",
    gpu: int = 0, 
    checkpoint_path: str = "./workdirs/MonoDTR/checkpoint/MonoDTR.pth",
):
    # Read Config
    cfg = cfg_from_file(config)
    
    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)

    cfg.is_running_test_set = True

    # Create StrongSORT model
    config_strongsort = './submodules/Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml'
    strong_sort_weights = "./weights/osnet_x0_25_msmt17.pt"  # model.pt path
    device = select_device("")
    half = False
    scfg = get_config()
    scfg.merge_from_file(config_strongsort)
    strongsort = StrongSORT(
        strong_sort_weights,
        device,
        half,
        max_dist=scfg.STRONGSORT.MAX_DIST,
        max_iou_distance=scfg.STRONGSORT.MAX_IOU_DISTANCE,
        max_age=scfg.STRONGSORT.MAX_AGE,
        n_init=scfg.STRONGSORT.N_INIT,
        nn_budget=scfg.STRONGSORT.NN_BUDGET,
        mc_lambda=scfg.STRONGSORT.MC_LAMBDA,
        ema_alpha=scfg.STRONGSORT.EMA_ALPHA,
    )
    strongsort.model.warmup()
    
    # Create detection the model
    detector = MonoDTR(cfg.detector)
    detector = detector.cuda()
    state_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
    new_dict = state_dict.copy()
    detector.load_state_dict(new_dict, strict=False)
    detector.eval()


    # Run evaluation
    dataset = NuScenesMonoDataset(cfg, VIDEO_DIR)
    with torch.no_grad():
        detector.eval()
        result_path = os.path.join(cfg.path.preprocessed_path, 'data')
        if os.path.isdir(result_path):
            os.system("rm -r {}".format(result_path))
            print("clean up the recorder directory of {}".format(result_path))
        os.mkdir(result_path)
        print("rebuild {}".format(result_path))

        projector = BBox3dProjector().cuda()
        backprojector = BackProjection().cuda()

        detections: "List[Detection]" = []
        for index in tqdm(range(len(dataset))):
            data = dataset[index]
            if isinstance(data['calib'], list):
                P2 = data['calib'][0]
            else:
                P2 = data['calib']
            collated_data = dataset.collate_fn([data])
                

            scores, bbox, obj_names = test_mono_detection(collated_data, model, None, cfg=cfg)
            assert scores.shape[0] == bbox.shape[0] and scores.shape[0] == len(obj_names)
            bbox_2d = bbox[:, 0:4]

            if bbox.shape[1] <= 4:
                raise Exception('Should run 3D')
            bbox_3d_state = bbox[:, 4:]  # [cx,cy,z,w,h,l,alpha, bot, top]
            bbox_3d_state_3d = backprojector(bbox_3d_state, P2)  # [x, y, z, w,h ,l, alpha, bot, top]

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

            detection = format_detections(scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)
            detections.append(detection)

        trackings = {}
        for detection in detections:
            bbox_2d = detection.bbox_2d
        print(sum(map(lambda x: x[1], detections)))


if __name__ == '__main__':
    fire.Fire(main)
