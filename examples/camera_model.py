"""Template: wrap a camera-conditioned world model.

Replace ``load_pipeline`` and the call inside ``generate_with_poses`` with your
model. Everything else (case loading, windows, trimming, video export,
evaluation manifest) is provided by ``src.models``.

    python generate.py --model examples.camera_model:MyCameraModel \
        --dataset /local/orbia-data --output /local/generations/my-camera-model
"""
import numpy as np

from src.models import CameraModel, camera


def load_pipeline(checkpoint, device):
    raise NotImplementedError('load your model here')


class MyCameraModel(CameraModel):
    name = 'my-camera-model'
    window_frames = 81          # frames per generation call; None for one call
    overlap_frames = 1          # shared boundary frames between calls
    resolution = (832, 480)     # model input (W, H)

    def __init__(self, checkpoint='/local/weights/my-model', device='cuda'):
        self.checkpoint, self.device = checkpoint, device
        self.pipeline = None

    def setup(self):
        self.pipeline = load_pipeline(self.checkpoint, self.device)

    def generate_with_poses(self, image, prompt, poses_c2w, K, *, case, history=None):
        # Resize/crop the conditioning image and keep the intrinsics consistent.
        h, w = image.shape[:2]
        K = camera.scale_intrinsics(K, (case.width, case.height), (w, h))
        image, K = camera.resize_and_crop(image, K, self.resolution)
        # Convert to the convention your model expects, e.g. OpenGL world-to-camera.
        w2c = camera.c2w_to_w2c(camera.opencv_to_opengl(poses_c2w)).astype(np.float32)
        frames = self.pipeline(image=image, prompt=prompt, extrinsics=w2c,
                               intrinsics=K.astype(np.float32), num_frames=len(poses_c2w))
        return list(frames)  # uint8 RGB [H, W, 3], one per pose
