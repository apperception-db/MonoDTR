import collections
import collections.abc
from datetime import datetime, timedelta
from typing import List, Optional

import cv2

import itertools
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

Float3 = Tuple[float, float, float]
Float4 = Tuple[float, float, float, float]
Float33 = Tuple[Float3, Float3, Float3]


def camera_config(
    camera_id: str,
    frame_id: str,
    frame_num: int,
    filename: str,
    camera_translation: "Float3",
    camera_rotation: "Float4",
    camera_intrinsic: "Float33",
    ego_translation: "Float3",
    ego_rotation: "Float4",
    timestamp: datetime,
    camera_heading: float,
    ego_heading: float,
    road_direction: float,
):
    _frame = CameraConfig()
    _frame.camera_id = camera_id
    _frame.frame_id = frame_id
    _frame.filename = filename
    _frame.timestamp = timestamp
    _frame._data = np.array(
        [
            frame_num,
            *camera_translation,
            *camera_rotation,
            *itertools.chain(*camera_intrinsic),
            *ego_translation,
            *ego_rotation,
            camera_heading,
            ego_heading,
            road_direction,
        ],
        dtype=np.float32,
    )
    return _frame


class CameraConfig:
    camera_id: str
    # TODO: remove
    frame_id: Optional[str]
    # TODO: remove
    filename: Optional[str]
    timestamp: datetime
    _data: "npt.NDArray[np.float32]"

    @property
    def frame_num(self) -> float:
        return self._data[0].item()

    @property
    def camera_translation(self) -> Float3:
        return self._data[1:4].tolist()

    @property
    def camera_rotation(self) -> Float4:
        rot = Quaternion(self._data[4:8]).unit
        return (rot[0], rot[1], rot[2], rot[3])

    @property
    def camera_intrinsic(self) -> Float33:
        return self._data[8:17].reshape((3, 3)).tolist()

    @property
    def ego_translation(self) -> Float3:
        return self._data[17:20].tolist()

    @property
    def ego_rotation(self) -> Float4:
        rot = Quaternion(self._data[20:24]).unit
        return (rot[0], rot[1], rot[2], rot[3])

    @property
    def camera_heading(self) -> float:
        return self._data[24].item()

    @property
    def ego_heading(self) -> float:
        return self._data[25].item()

    @property
    def road_direction(self) -> float:
        return self._data[26].item()

    def __iter__(self):
        yield self.camera_id
        yield self.frame_id
        yield self.frame_num
        yield self.filename
        yield self.camera_translation
        yield self.camera_rotation
        yield self.camera_intrinsic
        yield self.ego_translation
        yield self.ego_rotation
        yield self.timestamp
        yield self.camera_heading
        yield self.ego_heading
        yield self.road_direction


def interpolate(f1: CameraConfig, f2: CameraConfig, timestamp: datetime):
    t1 = f1.timestamp
    t2 = f2.timestamp
    total_delta = (t2 - t1).total_seconds()
    delta = (timestamp - t1).total_seconds()

    ratio = delta / total_delta

    _frame = CameraConfig()
    _frame.camera_id = f1.camera_id
    _frame.frame_id = None
    _frame.filename = None
    _frame.timestamp = timestamp
    _frame._data = (f2._data - f1._data) * ratio + f1._data

    return _frame



class Video(collections.abc.Iterable):
    videofile: str

    def __init__(
        self, videofile: str, camera_configs: "List[CameraConfig]", start: "datetime | None" = None
    ):
        self.videofile = videofile
        self._camera_configs: "List[CameraConfig]" = camera_configs
        self._start: "datetime | None" = start
        self._interpolated_frames: "List[CameraConfig] | None" = None
        self._num_frames: "int | None" = None
        self._fps: "float | None" = None

    @property
    def interpolated_frames(self):
        if self._interpolated_frames is None:
            num_frames, fps = self.__get_fps_and_num_frames()

            if len(self._camera_configs) == 1:
                self._start = self._camera_configs[0].timestamp
                self._interpolated_frames = [self._camera_configs[0] for _ in range(num_frames)]
            else:
                assert self._start is not None
                assert self._camera_configs[-1].timestamp > self._start + timedelta(
                    seconds=(num_frames - 1) / fps
                ), f"{self._camera_configs[-1].timestamp} {self._start + timedelta(seconds=(num_frames - 1) / fps)}"

                idx = 0
                self._interpolated_frames: "List[CameraConfig]" = []
                for i in range(num_frames):
                    t = self._start + timedelta(seconds=i / fps)
                    while self._camera_configs[idx + 1].timestamp < t:
                        idx += 1
                    self._interpolated_frames.append(
                        interpolate(self._camera_configs[idx], self._camera_configs[idx + 1], t)
                    )
        return self._interpolated_frames

    @property
    def fps(self):
        return self.__get_fps_and_num_frames()[1]

    def __getitem__(self, index):
        return self.interpolated_frames[index]

    def __iter__(self) -> "collections.abc.Iterator[CameraConfig]":
        return iter(self.interpolated_frames)

    def __len__(self):
        if self._interpolated_frames is not None:
            return len(self._interpolated_frames)

        return self.__get_fps_and_num_frames()[0]

    def __get_fps_and_num_frames(self):
        if self._num_frames is None or self._fps is None:
            cap = cv2.VideoCapture(self.videofile)
            self._num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            cv2.destroyAllWindows()
        return self._num_frames, self._fps
