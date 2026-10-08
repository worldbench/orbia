"""Template: wrap an image-and-text-to-video model or API.

Each request receives the scene caption plus a timed motion script for the
frames it generates. Continuations start from the last generated frame(s).

    python generate.py --model examples.text_model:MyTextModel \
        --dataset /local/orbia-data --output /local/generations/my-ti2v --horizon short

Inspect the prompts without running a model:

    python -c "from src.models import load_model_case, camera_text; \
c = load_model_case('/local/orbia-data', 'case_0001'); print(camera_text.camera_prompt(c, 24)['prompt'])"
"""
from src.models import TextModel


class MyTextModel(TextModel):
    name = 'my-ti2v-model'
    native_fps = 24.0        # prompt times follow the model's own clock
    segment_frames = 121     # frames per request; None for one request
    context_frames = 1       # conditioning frames carried into the next request
    overlap_frames = 1       # repeated frames removed at each boundary

    def generate_segment(self, context, prompt, num_frames, *, case, index):
        # context[-1] is the input image (index 0) or the last generated frame.
        raise NotImplementedError('call your image-to-video model or API here')
