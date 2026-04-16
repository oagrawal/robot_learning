"""
Interactive viewer to mark failure-region [start_t, end_t] for transition_val JSON.

Plays agentview in slow motion (optional autoplay) or step frame-by-frame, shows timestep,
and binds start/end per demo. Merges JSON on 'w' / 's' and on exit.

Usage:
    cd imitation/imitation
    python data/visualize_demo_labeling.py \\
        --input ./data/failure_dense_labeled_1000.hdf5 \\
        --output /abs/path/to/my_transition_val.json \\
        --file_idx 1

Keybinds (window must be focused):
    ,  or  k     Previous timestep
    .  or  l     Next timestep
    [            Set start_t = current t
    ]            Set end_t   = current t
    n            Next demo
    p            Previous demo
    w            Write this demo's region to JSON (needs start and end)
    s            Save full JSON (all regions in memory)
    r            Clear start/end for current demo only
    Space        Toggle slow autoplay (--fps)
    q  or  ESC   Quit and save full JSON

Requires a display (OpenCV highgui).

Inspired by visualize_rollout.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np


def _load_images(demo) -> np.ndarray:
    if "obs/agentview_image" in demo:
        return demo["obs/agentview_image"][:]
    if "obs" in demo and "agentview_image" in demo["obs"]:
        return demo["obs"]["agentview_image"][:]
    raise KeyError("No obs/agentview_image or obs['agentview_image'] in this demo")


def draw_frame(
    rgb: np.ndarray,
    *,
    demo_key: str,
    demo_index: int,
    n_demos: int,
    t: int,
    total_steps: int,
    start_t: Optional[int],
    end_t: Optional[int],
    file_idx: int,
    autoplay: bool,
    fps: float,
) -> np.ndarray:
    frame = rgb.copy()
    h, w = frame.shape[:2]
    scale = max(1, 480 // max(h, 1))
    frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    h, w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    y = 18
    dy = 18
    lines = [
        f"{demo_key}  ({demo_index + 1}/{n_demos})  file_idx={file_idx}",
        f"t = {t} / {total_steps - 1}   (len={total_steps})",
        f"start_t = {start_t if start_t is not None else '---'}   "
        f"end_t = {end_t if end_t is not None else '---'}",
        f"autoplay: {'ON' if autoplay else 'off'}  ({fps:.1f} fps)",
        "[ ] set start/end   w write demo   s save all   r clear   n/p demo   ,/. or k/l frame",
    ]
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += dy

    if start_t is not None and end_t is not None:
        lo, hi = min(start_t, end_t), max(start_t, end_t)
        if lo <= t <= hi:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 220, 0), 3)

    return frame


def load_json_regions(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return [], {}
    with open(path, "r") as f:
        data = json.load(f)
    regions = list(data.get("failure_regions", []))
    meta = {k: v for k, v in data.items() if k != "failure_regions"}
    return regions, meta


def save_json(
    path: str,
    regions: List[Dict[str, Any]],
    *,
    hdf5_path: str,
    file_idx: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    doc: Dict[str, Any] = {
        "version": 1,
        "description": "failure_regions: inclusive [start_t, end_t] per demo for hard val/test",
        "hdf5_path": os.path.abspath(hdf5_path),
        "file_idx": int(file_idx),
        "failure_regions": regions,
    }
    if extra_meta:
        for k, v in extra_meta.items():
            if k not in doc:
                doc[k] = v
    out_abs = os.path.abspath(path)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_abs, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Saved {len(regions)} region(s) -> {out_abs}")


def merge_region_for_demo(
    regions: List[Dict[str, Any]],
    file_idx: int,
    demo_key: str,
    start_t: int,
    end_t: int,
    notes: str = "",
) -> List[Dict[str, Any]]:
    lo, hi = min(start_t, end_t), max(start_t, end_t)
    new_entry = {
        "file_idx": int(file_idx),
        "demo_key": demo_key,
        "start_t": int(lo),
        "end_t": int(hi),
        "notes": notes,
    }
    out = [
        r
        for r in regions
        if not (r.get("demo_key") == demo_key and int(r.get("file_idx", -1)) == file_idx)
    ]
    out.append(new_entry)
    return out


def regions_to_per_demo(
    regions: List[Dict[str, Any]], file_idx: int
) -> Dict[str, Dict[str, Optional[int]]]:
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for r in regions:
        if int(r.get("file_idx", -1)) != file_idx:
            continue
        dk = r.get("demo_key")
        if not dk:
            continue
        out[str(dk)] = {"start": int(r["start_t"]), "end": int(r["end_t"])}
    return out


def run(args: argparse.Namespace) -> None:
    hdf5_path = os.path.abspath(os.path.expanduser(args.input))
    out_path = os.path.abspath(os.path.expanduser(args.output))
    file_idx = int(args.file_idx)

    regions, meta = load_json_regions(out_path)
    if regions:
        print(f"Loaded {len(regions)} existing region(s) from {out_path}")

    per_demo_state: Dict[str, Dict[str, Optional[int]]] = regions_to_per_demo(regions, file_idx)

    def get_state(dk: str) -> Tuple[Optional[int], Optional[int]]:
        st = per_demo_state.setdefault(dk, {"start": None, "end": None})
        return st["start"], st["end"]

    def set_start(dk: str, v: Optional[int]) -> None:
        per_demo_state.setdefault(dk, {"start": None, "end": None})["start"] = v

    def set_end(dk: str, v: Optional[int]) -> None:
        per_demo_state.setdefault(dk, {"start": None, "end": None})["end"] = v

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        n_demos = len(demos)
        if n_demos == 0:
            print("No demos in file.")
            return

        demo_idx = max(0, min(args.start_demo, n_demos - 1))
        t = 0
        autoplay = bool(args.autoplay)
        fps = max(0.5, float(args.fps))
        delay_ms = max(1, int(1000.0 / fps))

        win = "Demo labeling (transition regions)"

        while True:
            demo_key = demos[demo_idx]
            demo = f[f"data/{demo_key}"]
            images = _load_images(demo)
            total = len(images)
            t = max(0, min(t, total - 1))

            start_t, end_t = get_state(demo_key)

            frame = draw_frame(
                images[t],
                demo_key=demo_key,
                demo_index=demo_idx,
                n_demos=n_demos,
                t=t,
                total_steps=total,
                start_t=start_t,
                end_t=end_t,
                file_idx=file_idx,
                autoplay=autoplay,
                fps=fps,
            )
            cv2.imshow(win, frame)

            wait = delay_ms if autoplay else 0
            raw = cv2.waitKey(wait)
            if raw == -1:
                key = 0
            else:
                key = raw & 0xFF

            if autoplay and key == 0:
                t = (t + 1) % total

            if key == 27 or key == ord("q"):
                save_json(out_path, regions, hdf5_path=hdf5_path, file_idx=file_idx, extra_meta=meta)
                print("Quit.")
                break

            if key == ord(" ") or key == 32:
                autoplay = not autoplay
                print(f"Autoplay: {autoplay}")

            elif key == ord(",") or key == ord("k"):
                t = max(0, t - 1)
            elif key == ord(".") or key == ord("l"):
                t = min(total - 1, t + 1)

            elif key == ord("[") or key == ord("b"):
                set_start(demo_key, t)
                print(f"  {demo_key}: start_t = {t}")
            elif key == ord("]"):
                set_end(demo_key, t)
                print(f"  {demo_key}: end_t = {t}")

            elif key == ord("n"):
                demo_idx = min(n_demos - 1, demo_idx + 1)
                t = 0
                print(f"Demo -> {demos[demo_idx]}")
            elif key == ord("p"):
                demo_idx = max(0, demo_idx - 1)
                t = 0
                print(f"Demo -> {demos[demo_idx]}")

            elif key == ord("r"):
                set_start(demo_key, None)
                set_end(demo_key, None)
                print(f"  Cleared start/end for {demo_key}")

            elif key == ord("w"):
                st, en = get_state(demo_key)
                if st is None or en is None:
                    print("  Set both start ([) and end (]) before w.")
                else:
                    regions = merge_region_for_demo(regions, file_idx, demo_key, st, en)
                    lo, hi = min(st, en), max(st, en)
                    set_start(demo_key, lo)
                    set_end(demo_key, hi)
                    save_json(out_path, regions, hdf5_path=hdf5_path, file_idx=file_idx, extra_meta=meta)
                    print(f"  Recorded region for {demo_key}")

            elif key == ord("s"):
                save_json(out_path, regions, hdf5_path=hdf5_path, file_idx=file_idx, extra_meta=meta)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label failure transition regions for JSON")
    parser.add_argument("--input", type=str, required=True, help="Path to HDF5 (e.g. failures)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <input_dir>/<basename>_transition_regions.json)",
    )
    parser.add_argument(
        "--file_idx",
        type=int,
        default=1,
        help="file_idx in each failure_regions entry (ClassifierDataset data_paths index)",
    )
    parser.add_argument("--start_demo", type=int, default=0, help="Initial demo index (0-based)")
    parser.add_argument("--fps", type=float, default=4.0, help="Autoplay FPS when Space is ON")
    parser.add_argument("--autoplay", action="store_true", help="Start with autoplay ON")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(os.path.expanduser(args.input))),
            f"{base}_transition_regions.json",
        )

    run(args)


if __name__ == "__main__":
    main()
