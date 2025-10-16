#!/usr/bin/env python3
import os
import sys
import argparse
import joblib
import numpy as np


def coerce_fps(fps_val):
    if isinstance(fps_val, (int, float)):
        return float(fps_val)
    # numpy scalar or single-element array
    if hasattr(fps_val, 'item'):
        try:
            return float(fps_val.item())
        except Exception:
            pass
    return float(fps_val)


def slice_first_dim(arr, new_len):
    try:
        a = np.asarray(arr)
        if a.ndim >= 1 and a.shape[0] >= new_len:
            return a[:new_len]
        return a
    except Exception:
        return arr


def main():
    parser = argparse.ArgumentParser(description="Trim a motion pkl to the first N seconds.")
    parser.add_argument('input_pkl', type=str, help='Path to input pkl')
    parser.add_argument('seconds', type=float, help='Seconds to keep from start')
    parser.add_argument('--out', type=str, default='', help='Output pkl path (default: <input>_first<seconds>s.pkl)')
    args = parser.parse_args()

    in_path = args.input_pkl
    seconds = max(0.0, float(args.seconds))
    if not os.path.exists(in_path):
        print(f"Error: input file not found: {in_path}")
        sys.exit(1)

    data = joblib.load(in_path)
    if not isinstance(data, dict) or len(data) == 0:
        print("Error: unexpected pkl format (expect dict with one motion entry)")
        sys.exit(1)

    key = next(iter(data.keys()))
    motion = data[key]
    if 'dof' not in motion or 'fps' not in motion:
        print("Error: motion dict missing required keys 'dof' and/or 'fps'")
        sys.exit(1)

    fps = coerce_fps(motion['fps'])
    orig_len = int(np.asarray(motion['dof']).shape[0])
    keep_frames = int(round(seconds * fps))
    keep_frames = max(0, min(keep_frames, orig_len))

    print(f"Input: {in_path}")
    print(f"fps={fps}, orig_frames={orig_len}, keep_frames={keep_frames}")

    # Create trimmed copy
    trimmed = {}
    for k, v in motion.items():
        # Slice arrays whose first dimension matches orig_len
        try:
            a = np.asarray(v)
            if a.ndim >= 1 and a.shape[0] == orig_len and keep_frames < orig_len:
                trimmed[k] = slice_first_dim(a, keep_frames)
            else:
                trimmed[k] = v
        except Exception:
            trimmed[k] = v

    out_data = {key: trimmed}

    out_path = args.out if args.out else (
        os.path.splitext(in_path)[0] + f"_first{int(seconds)}s.pkl"
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(out_data, out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()


