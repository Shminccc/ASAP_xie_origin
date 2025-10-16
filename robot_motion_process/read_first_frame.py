#!/usr/bin/env python3
import sys
import os
import joblib
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python robot_motion_process/read_first_frame.py <path_to_pkl>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    if not os.path.exists(pkl_path):
        print(f"Error: file not found: {pkl_path}")
        sys.exit(1)

    data = joblib.load(pkl_path)
    # pkl 结构为 { PosixPath(...npz): motion_dict }
    first_key = next(iter(data.keys()))
    motion = data[first_key]

    def fmt(arr):
        arr = np.asarray(arr)
        return np.array2string(arr, precision=6, suppress_small=False, threshold=200)

    print(f"File: {pkl_path}")
    print(f"Motion key: {first_key}")
    # 基本字段
    fps = motion.get('fps', None)
    print(f"fps: {fps}")

    # 根位姿与第一帧 DOF
    if 'root_trans_offset' in motion:
        print("root_trans_offset[0]:", fmt(motion['root_trans_offset'][0]))
    if 'root_rot' in motion:
        print("root_rot[0] (xyzw):", fmt(motion['root_rot'][0]))
    if 'dof' in motion:
        print("dof[0] (first frame):", fmt(motion['dof'][0]))

    # 可选字段
    if 'pose_aa' in motion:
        print("pose_aa[0] shape:", np.asarray(motion['pose_aa']).shape)
    if 'contact_mask' in motion:
        print("contact_mask[0]:", fmt(motion['contact_mask'][0]))


if __name__ == "__main__":
    main()


