"""Scene-only VLM review applied after anchor and trajectory checks."""
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


HAZARDS = (
    "shot_boundary_or_no_coview",
    "central_or_prominent_people",
    "dynamic_content_dominant",
    "low_light_or_visibility_loss",
    "severe_weather_or_lens_contamination",
    "subtitle_or_watermark",
    "open_or_nature_dominated",
)

SCENE_REVIEW_PROMPT = """Review the five ordered video frames: Q0, 25%,
middle, 75%, and QE.

First classify the scene as indoor or outdoor. Then decide whether the scene
should be accepted or rejected.

Reject the scene if any of the following is true:

- The frames cross a shot or editing boundary, or do not show one continuous scene.
- One or more people are central, prominent, being followed, presenting to the
  camera, or visibly moving through the main part of the scene across multiple
  frames. A single such person is enough to reject the scene, even when the
  surrounding room or building occupies more pixels.
- Moving people, vehicles, or other dynamic content strongly interfere with
  observation of the static scene.
- The image is materially dark, blurred, foggy, obscured, or difficult to inspect.
- Heavy rain, snow, lens contamination, glare, or other weather significantly
  reduces visibility.
- Any subtitle, channel name, logo, or watermark is overlaid on the video.
  Carefully inspect all four corners in every frame. Fixed screen-position text
  is an overlay even when it is small, faint, or semi-transparent. Physical text
  on signs, products, or buildings is not an overlay.
- The scene is mainly open or natural scenery such as road, ground, snow, water,
  mountain, forest, or generic vegetation, without clear distinctive built objects.

Do not reject only because:
- one or two people remain small and peripheral in every frame;
- a vehicle is clearly parked;
- physical text appears on an object or building;
- vegetation appears around an otherwise distinctive built scene.

If any rejection condition is present, set scene_status to "reject".
Otherwise set it to "pass".

Return exactly one JSON object and no markdown:
{
  "environment": "indoor",
  "environment_confidence": "high",
  "scene_hazards": {
    "shot_boundary_or_no_coview": false,
    "central_or_prominent_people": false,
    "dynamic_content_dominant": false,
    "low_light_or_visibility_loss": false,
    "severe_weather_or_lens_contamination": false,
    "subtitle_or_watermark": false,
    "open_or_nature_dominated": false
  },
  "scene_status": "pass",
  "reason": "brief evidence-based reason"
}
"""


@dataclass(frozen=True)
class SceneReview:
    environment: str
    environment_confidence: str
    scene_hazards: Mapping[str, bool]
    scene_status: str
    reason: str

    @property
    def passed(self) -> bool:
        return self.scene_status == "pass"

    @property
    def flagged(self) -> list[str]:
        return [key for key in HAZARDS if self.scene_hazards[key]]


def parse_scene_review(text: str) -> SceneReview:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value: Any = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("scene review must be one JSON object")
    hazards = value.get("scene_hazards")
    if not isinstance(hazards, Mapping) or set(hazards) != set(HAZARDS):
        raise ValueError("scene_hazards keys do not match the prompt")
    if any(type(hazards[key]) is not bool for key in HAZARDS):
        raise ValueError("every scene hazard must be boolean")
    environment = str(value.get("environment"))
    if environment not in {"indoor", "outdoor"}:
        raise ValueError("environment must be indoor or outdoor")
    # The decision follows the hazards, so an inconsistent status cannot pass a flagged scene.
    status = "reject" if any(hazards.values()) else "pass"
    return SceneReview(environment, str(value.get("environment_confidence", "")),
                       {key: hazards[key] for key in HAZARDS}, status, str(value.get("reason", "")).strip())
