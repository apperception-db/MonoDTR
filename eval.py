import pickle
import fire
import torch
from copy import deepcopy
import numpy as np
from aputils import Video, camera_config
import cv2

from scripts._path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.data.pipeline import build_augmentator
from predict import predict

print('CUDA available: {}'.format(torch.cuda.is_available()))

VIDEO_DIR = "./data/boston-seaport/"


def main(
    config:str="config/config.py",
    gpu:int=0, 
    checkpoint_path:str="retinanet_79.pth",
):
    # Read Config
    cfg = cfg_from_file(config)
    
    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)

    cfg.is_running_test_set = True
    # Create the model
    detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
    detector = detector.cuda()

    state_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
    new_dict = state_dict.copy()
    detector.load_state_dict(new_dict, strict=False)
    detector.eval()

    # Run evaluation
    predict(cfg, detector, NuScenesMonoDataset(cfg, VIDEO_DIR), None, 0)
    print('finish')


class NuScenesMonoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, video_dir: str):
        self.output_dict = {
                "calib": False,
                "image": True,
                "label": False,
                "velodyne": False
            }
        self.transform = build_augmentator(cfg.data.test_augmentation)
        self.obj_types = cfg.obj_types

        scene = "scene-0757-CAM_FRONT"
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


if __name__ == '__main__':
    fire.Fire(main)
