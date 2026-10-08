"""Templates: wrap an action-conditioned world model.

Two layouts are supported:

* ``BlockActionModel``: a programmatic streaming model that consumes one
  ``W A S D I J K L`` vector per generation block.
* ``TimedActionModel``: an interactive model (SDK or browser) driven by
  held keys on a wall-clock schedule.

Calibrate the turning response first: generate a few clips holding a single
look key, recover cameras with DA3 (``evaluate.py --stage geometry``),
and measure degrees per block or degrees per second.
"""
import numpy as np

from src.models import ActionModel
from src.models.actions import BlockActionSpec, TimedActionSpec


class BlockActionModel(ActionModel):
    name = 'my-block-action-model'
    action_mode = 'block'
    block_spec = BlockActionSpec(first_block_frames=9, block_frames=12,
                                 yaw_step_deg=20.0, pitch_step_deg=17.0)

    def setup(self):
        self.stream = None  # load your streaming model here

    def generate_with_actions(self, image, prompt, plan, *, case):
        # plan['actions'][k] is the 8-dim vector for block k, ordered plan['keys'].
        frames = []
        state = self.stream.start(image=image, prompt=prompt)
        for vector in plan['actions']:
            frames.extend(self.stream.step(state, np.asarray(vector, dtype=np.float32)))
        return frames


class TimedActionModel(ActionModel):
    name = 'my-timed-action-model'
    action_mode = 'timed'
    timed_spec = TimedActionSpec(yaw_deg_per_s=70.0, pitch_deg_per_s=40.0,
                                 control_seconds={'short': 20.0, 'long': 54.0})

    def generate_with_actions(self, image, prompt, plan, *, case):
        # plan['intervals'] = [{'start_s', 'end_s', 'keys'}]; press/release keys at
        # these absolute times, record the stream, then return the captured frames.
        # The runner keeps the first case.num_frames frames.
        raise NotImplementedError('dispatch plan["intervals"] to your interactive session')
