"""Sphere-trace planned camera paths before rendering."""

import json
import math
import os
from itertools import pairwise
from pathlib import Path

import unreal

CAMERA_LABELS = ("ORBIA_Camera",)


def _ignored_actors():
    """Ignore any camera actor left in the level by an earlier run."""
    actors = unreal.get_editor_subsystem(
        unreal.EditorActorSubsystem
    ).get_all_level_actors()
    return [
        actor
        for actor in actors
        if actor.get_actor_label() in CAMERA_LABELS
        or isinstance(actor, unreal.CineCameraActor)
    ]


def _assert_world_is_loaded(actor_count, cases):
    """Reject an empty or unstreamed map without assuming a ground plane."""
    if actor_count <= 0:
        raise RuntimeError("the loaded level contains no actors")
    labels = set(cases[0].get("anchor_actor_labels") or [])
    labels.add(cases[0].get("anchor_actor_label", ""))
    labels.discard("")
    actor_labels = {
        actor.get_actor_label()
        for actor in unreal.get_editor_subsystem(
            unreal.EditorActorSubsystem
        ).get_all_level_actors()
    }
    missing = sorted(labels - actor_labels)
    if missing:
        raise RuntimeError(
            "requested anchor actor(s) are not resident in the loaded map: "
            + ", ".join(missing)
        )


def _hit_payload(hit):
    """Describe a blocking collision hit."""
    if hit is None:
        return None
    component = getattr(hit, "component", None)
    actor = component.get_owner() if component is not None else None
    return {
        "blocking_hit": bool(getattr(hit, "blocking_hit", False)),
        "initial_overlap": bool(getattr(hit, "start_penetrating", False)),
        "actor_label": actor.get_actor_label() if actor is not None else None,
        "actor_path": actor.get_path_name() if actor is not None else None,
        "component": component.get_name() if component is not None else None,
    }


def _probe(world, locations, radius_cm, ignored):
    """Return the first colliding segment and how far along the path it is."""
    origin = locations[0]
    collisions = []
    for index, (start, end) in enumerate(pairwise(locations)):
        if math.dist(start, end) < 0.001:
            continue
        # Trace on the visibility channel; object-type filtering misses some blocking actors.
        raw = unreal.SystemLibrary.sphere_trace_single(
            world,
            unreal.Vector(*start),
            unreal.Vector(*end),
            radius_cm,
            unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            False,
            ignored,
            unreal.DrawDebugTrace.NONE,
            True,
        )
        hit = _hit_payload(raw[1] if isinstance(raw, tuple) and len(raw) == 2 else raw)
        if hit is not None and (hit["blocking_hit"] or hit["initial_overlap"]):
            collisions.append(
                {
                    "frame": index,
                    **hit,
                }
            )
            if len(collisions) >= 10:
                break
    if not collisions:
        clear_to = max(
            math.dist(origin[:2], point[:2]) for point in locations
        )
        return {
            "status": "pass",
            "collision_count_reported": 0,
            "first_collision_frame": None,
            "clear_horizontal_travel_m": clear_to / 100.0,
        }
    first = collisions[0]["frame"]
    reached = max(
        math.dist(origin[:2], locations[index][:2]) for index in range(first + 1)
    )
    return {
        "status": "fail",
        "collision_count_reported": len(collisions),
        "first_collision_frame": first,
        "first_collision": collisions[0],
        "clear_horizontal_travel_m": reached / 100.0,
    }


def main():
    plan_index = Path(os.environ["ORBIA_PLAN_INDEX"])
    output = Path(os.environ["ORBIA_PROBE_OUTPUT"])
    radius_cm = float(os.environ.get("ORBIA_CAMERA_RADIUS_CM", "15.0"))
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    ignored = _ignored_actors()
    cases = json.loads(plan_index.read_text(encoding="utf-8"))["cases"]

    actor_count = len(
        unreal.get_editor_subsystem(
            unreal.EditorActorSubsystem
        ).get_all_level_actors()
    )
    _assert_world_is_loaded(actor_count, cases)
    unreal.log(f"ORBIA_PROBE_WORLD_READY actors={actor_count}")

    results = []
    for position, case in enumerate(cases, start=1):
        spec = json.loads(Path(case["trajectory"]).read_text(encoding="utf-8"))
        locations = [
            [float(value) for value in frame["ue_world"]["location_cm"]]
            for frame in spec["frames"]
        ]
        report = _probe(world, locations, radius_cm, ignored)
        report.update(
            {
                "case_id": case.get("case_id", spec.get("case_id")),
                "source_case_id": (
                    (case.get("contextual_search") or {}).get("source_case_id")
                    or case.get("case_id")
                    or spec.get("case_id")
                ),
                "scene": case.get("scene", ""),
                "template": case["template"],
                
                "trajectory_summary": {
                    "angular_span_deg": spec.get("derived", {}).get(
                        "angular_span_deg"
                    ),
                    "maximum_displacement_m": spec.get("derived", {}).get(
                        "maximum_displacement_m"
                    ),
                    "arc_offset_m": spec.get("derived", {}).get("arc_offset_m"),
                },
                "requested_travel_m": max(
                    math.dist(locations[0][:2], point[:2]) for point in locations
                )
                / 100.0,
            }
        )
        results.append(report)
        unreal.log(
            "ORBIA_PROBE {0}/{1} {2}/{3} {4} clear_to={5:.2f}m requested={6:.2f}m".format(
                position,
                len(cases),
                report["scene"],
                report["template"],
                report["status"],
                report["clear_horizontal_travel_m"],
                report["requested_travel_m"],
            )
        )

    failed = [item for item in results if item["status"] != "pass"]
    payload = {
        "contract": "orbia.unreal.clearance-probe.v1",
        "note": (
            "sphere sweeps of requested camera poses"
        ),
        "camera_radius_cm": radius_cm,
        "level_actor_count": actor_count,
        "case_count": len(results),
        "passed_case_count": len(results) - len(failed),
        "failed_case_count": len(failed),
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    unreal.log(
        f"ORBIA_PROBE_FINISHED passed={payload['passed_case_count']} "
        f"failed={payload['failed_case_count']} output={output}"
    )
    # Set ORBIA_QUIT_EDITOR=1 to close the editor after a headless run.
    if os.environ.get("ORBIA_QUIT_EDITOR", "0") == "1":
        unreal.SystemLibrary.quit_editor()


if __name__ == "__main__":
    main()
