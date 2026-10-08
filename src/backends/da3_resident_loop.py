"""Official SALAD loop detector with resident weights and shared RGB input."""
from unittest.mock import patch


def resident_loop_detector(base_class, dino_repo):
    import torch
    from PIL import Image

    weights = {}

    class ResidentLoopDetector(base_class):
        def load_model(self):
            if not weights:
                original = torch.hub.load
                def local_backbone(repo, model, *args, **kwargs):
                    if repo != 'facebookresearch/dinov2':
                        return original(repo, model, *args, **kwargs)
                    # SALAD loads the full state_dict strictly immediately after
                    # construction, so a separate pretrained backbone is unnecessary.
                    kwargs.update(source='local', pretrained=False)
                    return original(str(dino_repo), model, *args, **kwargs)
                with patch.object(torch.hub, 'load', local_backbone):
                    model, device = super().load_model()
                weights.update(model=model, device=device)
                print('SALAD_MODEL_LOAD_COUNT=1', flush=True)
            self.model, self.device = weights['model'], weights['device']
            return self.model, self.device

        def get_image_paths(self):
            self.image_paths = [f'frame_{i:06d}' for i in range(len(self.rgb_frames))]
            return self.image_paths

        def extract_descriptors(self):
            transform = self._input_transform(self.image_size)
            descriptors = []
            for start in range(0, len(self.rgb_frames), self.batch_size):
                batch = torch.stack([
                    transform(Image.fromarray(rgb))
                    for rgb in self.rgb_frames[start:start + self.batch_size]
                ]).to(self.device)
                with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    descriptors.append(self.model(batch).cpu())
            self.descriptors = torch.cat(descriptors)
            return self.descriptors

    return ResidentLoopDetector
