import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List
import sys
from pathlib import Path

import cv2
import fire
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data
from tqdm import tqdm

from aputils import Video, camera_config
from scripts._path_init import *
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.networks.detectors.monodtr_detector import MonoDTR
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector
from visualDet3D.utils.utils import cfg_from_file

if "./submodules" not in sys.path:
    sys.path.append("./submodules")

FILE = Path(__file__).resolve()
SUBMODULE = FILE.parent / "submodules"
YOLOV5_STRONGSORT_OSNET_ROOT = SUBMODULE / "Yolov5_StrongSORT_OSNet"

if str(YOLOV5_STRONGSORT_OSNET_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT))
if str(YOLOV5_STRONGSORT_OSNET_ROOT / "trackers") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT / "trackers"))
if str(YOLOV5_STRONGSORT_OSNET_ROOT / "trackers/strong_sort") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT / "trackers/strong_sort"))
if str(YOLOV5_STRONGSORT_OSNET_ROOT / "trackers/strong_sort/deep/reid") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSNET_ROOT / "trackers/strong_sort/deep/reid"))

from submodules.Yolov5_StrongSORT_OSNet.trackers.multi_tracker_zoo import create_tracker
from submodules.Yolov5_StrongSORT_OSNet.yolov5.utils.general import scale_boxes
from submodules.Yolov5_StrongSORT_OSNet.yolov5.utils.plots import Annotator, colors
from submodules.Yolov5_StrongSORT_OSNet.trackers.strong_sort.strong_sort import StrongSORT

print('CUDA available: {}'.format(torch.cuda.is_available()))

VIDEO_DIR = "./data/boston-seaport/"

names = ['Car', 'Pedestrian', 'Cyclist']


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
        self._images = []
        self.cap = cv2.VideoCapture(os.path.join(video_dir, videofile))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self):
        return len(self.video)
    
    def get_image(self, index: "int"):
        if index >= self.frame_count:
            raise IndexError()

        while index >= len(self._images):
            ret, frame = self.cap.read()
            if not ret:
                raise Exception()
            self._images.append(frame)
        
        if len(self._images) == self.frame_count and self.cap.isOpened():
            assert len(self._images) == len(self.video), (len(self._images), len(self.video))
            ret, frame = self.cap.read()
            assert not ret
            self.cap.release()
            cv2.destroyAllWindows()
        
        return self._images[index]

    def __getitem__(self, index: "int"):
        if index >= self.frame_count:
            raise IndexError()

        frame = self.video[index]
        image = torch.tensor(self.get_image(index)[:, :, ::-1].copy())
        p2 = torch.tensor(np.concatenate([
            np.array(frame.camera_intrinsic),
            np.array([[0, 0, 0]]).T
        ], axis=1))
        transformed_image, transformed_P2 = self.transform(image, p2=deepcopy(p2))

        output_dict = {
            'calib': transformed_P2,
            'image': transformed_image,
            'original_shape':image.shape,
            'original_P':p2.copy(),
            'original_image': image,
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


def main(
    config: str = "config/config.py",
    gpu: int = 0, 
    checkpoint_path: str = "./workdirs/MonoDTR/checkpoint/MonoDTR.pth",
):
    # Read Config
    cfg: "Any" = cfg_from_file(config)
    
    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)

    cfg.is_running_test_set = True

    # Create StrongSORT model
    reid_weights = FILE.parent / "weights/osnet_x0_25_msmt17.pt"  # model.pt path
    device = torch.device(int(gpu))
    half = False
    tracker = create_tracker("strongsort", reid_weights, device, half)
    tracker.model.warmup()
    print('loaded tracker')
    
    # Create detection the model
    detector = MonoDTR(cfg.detector)
    detector = detector.cuda()
    state_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
    new_dict = state_dict.copy()
    detector.load_state_dict(new_dict, strict=False)
    detector.eval()
    print('loaded detector')

    video_writer = None

    # Run evaluation
    dataset = NuScenesMonoDataset(cfg, VIDEO_DIR)
    print('constructed dataset')
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

        trackings = {}
        output = None
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frame, prev_frame = None, None
        detections: "List[Detection]" = []
        for index in tqdm(range(len(dataset))):
            print(index)
            data = dataset[index]
            if isinstance(data['calib'], list):
                P2 = data['calib'][0]
            else:
                P2 = data['calib']
            collated_data = dataset.collate_fn([data])
                

            # images: torch.Tensor [N=1 x 3 x h x w]
            # P2: [np.array[3 x 4]]
            images, P2 = collated_data
            scores, bbox, clss = detector([
                images.cuda().float().contiguous(),
                torch.tensor(P2).cuda().float()
            ])
            print(scores)
            print(bbox)
            print(clss)
            # scores, bbox, obj_names = test_mono_detection(collated_data, detector, None, cfg=cfg)
            assert scores.shape[0] == bbox.shape[0] and scores.shape[0] == clss.shape[0]
            bbox_2d = bbox[:, 0:4]

            if bbox.shape[1] <= 4:
                raise Exception('Should run 3D')
            bbox_3d_state = bbox[:, 4:]  # [cx,cy,z,w,h,l,alpha, bot, top]
            P2 = P2[0]
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

            detection = format_detections(scores, bbox_2d, bbox_3d_state_3d, thetas, clss)
            detections.append(detection)

            im = collated_data[0]
            curr_frame = im
            im0 = data['original_image'].copy()
            print(im0.shape)
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if prev_frame is not None and curr_frame is not None:
                if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                    tracker.tracker.camera_update(prev_frame, curr_frame)
            det = bbox_2d
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                outputs = tracker.update(det.cpu(), im0)
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
                        bboxes = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])
                        index = int(output[7])
                        bbox_3d = bbox_3d_state_3d[index]

                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        label = f'{id} {names[cls]} {conf:.2f}'
                        annotator.box_label(bboxes, label, colod=colors(cls, True))
            
            im0 = annotator.result()

            if video_writer is None:
                fps = 20
                h = im0.shape[0]
                w = im0.shape[1]
                video_writer = cv2.VideoWriter('./test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            video_writer.write(im0)
            
            prev_frame = curr_frame
        
    video_writer.release()
    cv2.destroyAllWindows()
                

if __name__ == '__main__':
    fire.Fire(main)
