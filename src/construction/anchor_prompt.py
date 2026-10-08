"""Five-frame VLM scene gate and anchor proposals with short SAM3.1 noun prompts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "orbia.anchor_review.v1"
SCENE_HAZARDS = (
    "shot_boundary",
    "endpoint_no_coview",
    "low_light_or_blur",
    "severe_weather_or_visibility_loss",
    "salient_person_or_crowd",
    "moving_traffic_dominant",
    "persistent_text_overlay",
    "open_or_generic_scene",
    "no_distinctive_shared_anchor",
    "severe_anchor_occlusion",
)
SCENE_HAZARD_REASONS = {
    "shot_boundary": "multi_shot",
    "endpoint_no_coview": "no_coview",
    "low_light_or_blur": "unreliable_visual_evidence",
    "severe_weather_or_visibility_loss": "severe_weather",
    "salient_person_or_crowd": "dynamic_dominant",
    "moving_traffic_dominant": "dynamic_dominant",
    "persistent_text_overlay": "unreliable_visual_evidence",
    "open_or_generic_scene": "no_distinctive_anchor",
    "no_distinctive_shared_anchor": "no_distinctive_anchor",
    "severe_anchor_occlusion": "severe_occlusion",
}
ANCHOR_KINDS = frozenset({"instance", "compact_structure"})
VLM_SHORT_ANCHOR_PROMPT = """You are selecting scenes and visual anchors for ORBIA.

Task purpose
ORBIA evaluates whether an image-conditioned video world model preserves
a stable scene under controlled camera motion. The model starts from Q0 and
follows a planned camera trajectory. Registered evidence at Q0 and QE is used
to test whether the model preserves the same static physical content across
viewpoint changes. Visual anchors define local regions where object identity,
appearance, structure, and spatial persistence can be evaluated more reliably
than with full-frame comparison.

Scene gate
Before looking for anchors, decide whether the pair is a clear, relatively
static source scene for this task. Fill every boolean in scene_hazards. If any
hazard is true, pair_usable must be false and candidates must be empty. Scene
quality has priority: never rescue a rejected scene with an incidental detail.

Inspect hazards before naming any anchor, in this order: (1) inspect the number,
position and prominence of visible people in both endpoints; (2) compare endpoint sharpness and exposure;
(3) inspect the image for persistent subtitle lines or editorial text bands;
(4) decide whether the stable evidence is mainly natural/open scenery. Any
positive hazard must be recorded even when attractive static architecture is
also present.

Set the corresponding hazard true when Q0 and QE cross a shot boundary or lack
shared scene content; when darkness, blur, severe weather, or poor visibility
makes boundaries unreliable, including when one endpoint is materially softer
or more motion-blurred than the other; when a person is prominent, near the image
center or main task region, or visibly walking across it; or when a dense/large
crowd dominates the scene. A few small or peripheral background pedestrians are
allowed when they do not overlap the prospective anchors or create ambiguity.
Also reject when moving traffic or other dynamic
ambiguity interferes with either endpoint, even if static background structures exist;
when persistent subtitles, sentence-like caption lines, or an editorial text band
is overlaid on the video, especially across the bottom of the frame. Any
sentence-style subtitle is a hard rejection regardless of its size, transparency,
or whether it covers the proposed anchor; ordinary text physically present on a
real sign or storefront is not an overlay;
when the view is open, generic, repetitive, or dominated by roads, water,
vegetation, rocks, mountains, snow, or distant scenery without a prominent
bounded man-made target; or when every possible shared target is severely
occluded. These are general decision principles, not an exhaustive catalogue
of scene categories. A small incidental background object cannot rescue an
otherwise unsuitable scene.

Trees, shrubs, and other vegetation may appear in the background of an otherwise
usable scene, but they must never be proposed as anchors. If trees or natural
vegetation are the only plausible stable shared targets, set open_or_generic_scene
or no_distinctive_shared_anchor true and reject the pair. Natural terrain and
unbounded surfaces are also invalid anchors. In particular, do not propose a
road, pavement, floor, wall, railway surface or track, generic water, sky, or
amorphous background, even when it is large and clearly visible. Do not use one
of those surfaces as a SAM synonym for another target. Bounded facades, doors,
signs, and buildings remain eligible when they are the actual target.

Anchor selection
Only for a usable scene, propose 3-8 visual anchors. A valid anchor must:
- refer to one static physical object or one coherent bounded structure;
- refer to the same physical instance at Q0 and QE;
- be clearly visible and independently identifiable at both endpoints;
- preferably be a substantial foreground or middle-distance target, rather
  than a small distant detail, while not occupying almost the entire image;
- look coherent and bounded enough to segment as one target;
- be distinctive enough not to be confused with nearby instances.

Reject a target that is dynamic, ambiguous, unbounded, severely occluded, too
small or distant, easily interchangeable with repeated details, or not the same
physical instance at both endpoints. Trees and other vegetation are not valid
anchors even when one individual tree is prominent. Prefer larger, closer, more complete and
more distinctive targets. Do not judge mask connectivity, holes, or tracking;
downstream segmentation performs those checks.

Segmentation prompts
For each anchor, provide 2-4 synonymous noun phrases for SAM3.1. Each phrase
must contain 1-4 words, name the same physical target, and contain no spatial
directions, coordinates, boxes, mask IDs, or multiple semantic objects. Prompt
variants must remain at the same semantic target and scale: for example, a
viewing platform cannot use ground or dirt path as a synonym.

Output contract
Return exactly one JSON object and no other text. It must contain:
- schema_version: exactly "orbia.anchor_review.v1";
- scene_hazards: an object containing exactly these boolean fields:
  shot_boundary, endpoint_no_coview, low_light_or_blur,
  severe_weather_or_visibility_loss, salient_person_or_crowd,
  moving_traffic_dominant, persistent_text_overlay, open_or_generic_scene,
  no_distinctive_shared_anchor, severe_anchor_occlusion;
- pair_usable: boolean, true exactly when every scene hazard is false;
- pair_reason: one short explanation of the scene-level decision;
- candidates: an empty list for a rejected pair, otherwise 3-8 objects.

Each candidate object must contain only label, anchor_kind, and sam_prompts.
anchor_kind must be "instance" or "compact_structure". sam_prompts must contain
2-4 synonymous 1-4 word noun phrases for the same target. Do not output endpoint
descriptions, boxes, coordinates, mask IDs, markdown, or additional keys.
"""

_POSITION_WORDS = frozenset(
    {"left", "right", "upper", "lower", "overhead", "distant", "foreground", "background"}
)

_CATEGORY_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("doorway",), "doorway", "door"),
    (("front door", "glass door", "door"), "door", "doorway"),
    (("storefront",), "storefront", "facade"),
    (("facade",), "facade", "building"),
    (("window",), "window", "building window"),
    (("highway sign",), "highway sign", "road sign"),
    (("street sign", "store sign", "signage", "sign"), "sign", "road sign"),
    (("stone wall", "brick wall", "retaining wall", "wall"), "wall", "building wall"),
    (("overpass", "bridge", "walkway"), "bridge", "structure"),
    (("railway", "elevated train"), "railway", "bridge"),
    (("archway", "arch"), "arch", "doorway"),
    (("shutter",), "shutter", "window"),
    (("roof", "eaves"), "roof", "building roof"),
    (("balcony",), "balcony", "building"),
    (("house", "chalet"), "house", "building"),
    (("hotel", "building"), "building", "facade"),
    (("pillar", "column"), "pillar", "column"),
    (("railing",), "railing", "fence"),
    (("fence",), "fence", "railing"),
    (("umbrella",), "umbrella", "sunshade"),
    (("embankment", "rocks"), "rocks", "embankment"),
    (("hillside", "hill"), "hillside", "hill"),
    (("mountain", "ridge"), "mountain", "mountain ridge"),
    (("painting", "artwork", "wall art"), "painting", "wall art"),
    (("cabinet",), "cabinet", "cupboard"),
    (("street lamp", "lantern", "light fixture", "lamp"), "lamp", "light"),
    (("awning",), "awning", "canopy"),
    (("flag",), "flag", "flagpole"),
    (("planter",), "planter", "flower pot"),
    (("staircase", "steps"), "stairs", "steps"),
    (("batting cage", "netted structure", "cage"), "cage", "netted structure"),
    (("peninsula",), "peninsula", "landform"),
)

# Unbounded surfaces are also filtered here, independently of the prompt.
# A bounded target word overrides a contextual surface word: a ``road sign`` is
# a sign, while ``asphalt road`` is not an object anchor.
_UNBOUNDED_ANCHOR_TERMS = frozenset(
    {
        "road", "roads", "pavement", "pavements", "sidewalk", "asphalt", "driveway",
        "path", "walkway", "alley", "ground", "floor", "floors", "wall", "walls", "railway",
        "railways", "rail", "track", "tracks", "water", "river", "lake", "sea", "ocean",
        "sky", "background", "backgrounds", "backdrop", "terrain", "mountain", "mountains",
        "hillside", "hillsides", "slope", "slopes", "ridge", "ridges", "cliff", "cliffs",
        "shore", "riverbed",
    }
)
_BOUNDED_TARGET_TERMS = frozenset(
    {
        "facade", "façade", "door", "doorway", "sign", "signage", "building",
        "house", "storefront", "window", "arch", "archway", "bridge", "overpass",
        "station", "platform", "bench", "lamp", "lantern", "cabinet", "painting",
        "artwork", "sculpture", "statue", "fountain", "awning", "balcony", "pillar",
        "column", "railing", "fence", "umbrella", "planter", "staircase", "stairs",
    }
)


def passes_bounded_anchor_taxonomy(value: Any) -> bool:
    """Whether a label or SAM phrase names a bounded, eligible anchor."""

    words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(value).lower()))
    if not words:
        return False
    unbounded = words & _UNBOUNDED_ANCHOR_TERMS
    if not unbounded:
        return True
    return bool(words & _BOUNDED_TARGET_TERMS)


class SAM31PromptContractError(ValueError):
    """Raised when a VLM short-prompt response is unsafe or malformed."""


@dataclass(frozen=True)
class SAM31PromptCandidate:
    label: str
    sam_prompts: tuple[str, ...]
    anchor_kind: str = "instance"
    static: bool = True
    expected_q0: bool = True
    expected_qe: bool = True


@dataclass(frozen=True)
class SAM31PromptReview:
    pair_usable: bool
    pair_reject_reasons: tuple[str, ...]
    pair_reason: str
    candidates: tuple[SAM31PromptCandidate, ...]
    scene_failure_mode: str = "none"
    scene_hazards: tuple[tuple[str, bool], ...] = ()


def _normalize_sam_prompt(raw: Any, *, field: str) -> str:
    prompt = " ".join(str(raw).strip().lower().split())
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", prompt)
    if not 1 <= len(words) <= 4:
        raise SAM31PromptContractError(f"{field} prompts must contain 1-4 words")
    if "and" in words or "or" in words:
        raise SAM31PromptContractError(f"{field} prompts cannot join categories")
    if _POSITION_WORDS.intersection(words):
        raise SAM31PromptContractError(f"{field} prompts cannot contain spatial words")
    if not passes_bounded_anchor_taxonomy(prompt):
        raise SAM31PromptContractError(
            f"{field} prompts cannot name an unbounded surface or background"
        )
    return prompt


def normalize_sam_prompts(value: Any, *, field: str = "sam_prompts") -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SAM31PromptContractError(f"{field} must be a list")
    prompts: list[str] = []
    for raw in value:
        prompt = _normalize_sam_prompt(raw, field=field)
        if prompt not in prompts:
            prompts.append(prompt)
    if not 2 <= len(prompts) <= 4:
        raise SAM31PromptContractError(f"{field} must contain 2-4 unique prompts")
    return tuple(prompts)


def candidate_from_mapping(raw: Mapping[str, Any], *, field: str = "candidate") -> SAM31PromptCandidate:
    label = " ".join(str(raw.get("label", "")).strip().split())
    if not label:
        raise SAM31PromptContractError(f"{field}.label must be nonempty")
    if not passes_bounded_anchor_taxonomy(label):
        raise SAM31PromptContractError(
            f"{field}.label cannot name an unbounded surface or background"
        )
    anchor_kind = str(raw.get("anchor_kind", "instance"))
    if anchor_kind not in ANCHOR_KINDS:
        raise SAM31PromptContractError(f"{field}.anchor_kind is invalid")
    sam_prompts = normalize_sam_prompts(
        raw.get("sam_prompts"), field=f"{field}.sam_prompts"
    )
    return SAM31PromptCandidate(
        label=label,
        sam_prompts=sam_prompts,
        anchor_kind=anchor_kind,
        # The prompt already requires static targets visible at both endpoints.
        static=True,
        expected_q0=True,
        expected_qe=True,
    )


def parse_sam31_prompt_review_response(text: str) -> SAM31PromptReview:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise SAM31PromptContractError("response contains no JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise SAM31PromptContractError("response JSON is invalid") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise SAM31PromptContractError("response has an unexpected schema_version")
    raw_hazards = payload.get("scene_hazards")
    if not isinstance(raw_hazards, Mapping):
        raise SAM31PromptContractError("scene_hazards must be an object")
    if set(raw_hazards) != set(SCENE_HAZARDS):
        raise SAM31PromptContractError("scene_hazards must contain exactly the required fields")
    if not all(isinstance(raw_hazards[name], bool) for name in SCENE_HAZARDS):
        raise SAM31PromptContractError("scene_hazards values must be booleans")
    scene_hazards = tuple((name, raw_hazards[name]) for name in SCENE_HAZARDS)
    pair_usable = payload.get("pair_usable")
    if not isinstance(pair_usable, bool):
        raise SAM31PromptContractError("pair_usable must be boolean")
    expected_usable = not any(value for _, value in scene_hazards)
    if pair_usable != expected_usable:
        raise SAM31PromptContractError(
            "pair_usable must be true exactly when every scene hazard is false"
        )
    pair_reason = " ".join(str(payload.get("pair_reason", "")).strip().split())
    if not pair_reason:
        raise SAM31PromptContractError("pair_reason must be nonempty")
    active_hazards = [name for name, value in scene_hazards if value]
    failure_mode = active_hazards[0] if active_hazards else "none"
    reject_reasons = tuple(
        dict.fromkeys(SCENE_HAZARD_REASONS[name] for name in active_hazards)
    )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > 8:
        raise SAM31PromptContractError("candidates must be a list with at most 8 items")
    if not pair_usable:
        candidates = []
    if pair_usable and len(candidates) < 3:
        raise SAM31PromptContractError("usable pair needs at least 3 candidates")
    parsed_items: list[SAM31PromptCandidate] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            continue
        # Skip malformed prompt variants but keep the candidate.
        raw_prompts = raw.get("sam_prompts")
        if not isinstance(raw_prompts, list):
            raise SAM31PromptContractError(f"candidates[{index}].sam_prompts must be a list")
        valid_prompts: list[str] = []
        for prompt_index, prompt in enumerate(raw_prompts):
            try:
                normalized = _normalize_sam_prompt(
                    prompt, field=f"candidates[{index}].sam_prompts[{prompt_index}]"
                )
            except SAM31PromptContractError:
                continue
            if normalized not in valid_prompts:
                valid_prompts.append(normalized)
        if len(valid_prompts) < 2:
            continue
        try:
            parsed_items.append(
                candidate_from_mapping(
                    {**raw, "sam_prompts": valid_prompts}, field=f"candidates[{index}]"
                )
            )
        except SAM31PromptContractError:
            # Drop the candidate; fewer than three valid candidates rejects the
            # response below so that the VLM can be queried again.
            continue
    parsed = tuple(parsed_items)
    if pair_usable and not 3 <= len(parsed) <= 8:
        raise SAM31PromptContractError(
            "fewer than three candidates retain two valid direct prompts"
        )
    return SAM31PromptReview(
        pair_usable=pair_usable,
        pair_reject_reasons=reject_reasons,
        pair_reason=pair_reason,
        candidates=parsed,
        scene_failure_mode=failure_mode,
        scene_hazards=scene_hazards,
    )


