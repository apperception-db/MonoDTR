import os
import pickle
from copy import deepcopy

import cv2
import fire
import numpy as np
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

print('CUDA available: {}'.format(torch.cuda.is_available()))

VIDEO_DIR = "./data/boston-seaport/"


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
        return {
            "bbox_2d": bbox_2d,
            "obj_types": obj_types,
            "bbox_3d_state_3d": bbox_3d_state_3d,
            "thetas": thetas,
            "scores": scores,
        }
    return None


def test_one(
    cfg: "EasyDict",
    index: int,
    dataset: "torch.utils.data.Dataset",
    model: "nn.Module",
    backprojector: "BackProjection",
    projector: "BBox3dProjector"
):
    data = dataset[index]
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']
    collated_data = dataset.collate_fn([data])
        

    scores, bbox, obj_names = test_mono_detection(collated_data, model, None, cfg=cfg)
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

    return format_detections(scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)


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
    # Create the model
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

        detections = []
        for index in tqdm(range(len(dataset))):
            detections.append(test_one(cfg, index, dataset, detector, backprojector, projector))
        print(sum(map(lambda x: x[1], detections)))


if __name__ == '__main__':
    fire.Fire(main)
