import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1.astype(np.float64), point2.astype(np.float64))

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif keycode == 256 or chr(keycode) == "Q":
        print("Esc")
        sys.exit()
    elif chr(keycode) == "L":
        print("Faster")
        speed *= 1.5
    elif chr(keycode) == "K":
        print("Slower")
        speed /= 1.5
    elif chr(keycode) == "J":
        print("Rewind")
        rewind = not rewind
    elif keycode == 263: # left
        print("Prev Frame")
        time_step -= 1
        paused = True
    elif keycode == 262: # right
        print("Next Frame")
        time_step += 1
        paused = True


@hydra.main(version_base=None, config_path="../humanoidverse/config", config_name="base")
def main(cfg: DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave

    dt = 1 / 60.0  # 60FPS
    paused = False
    rewind = False
    motion_id = 0
    time_step = 0
    
    motion_file = cfg.motion_file
    speed = cfg.speed if 'speed' in cfg else 1.0
    
    print(motion_file)
    print("Motion file: ", motion_file)
    
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    num_motions = len(motion_data_keys)
    
    curr_motion = motion_data[motion_data_keys[motion_id]]
    print("Motion length: ", motion_data[motion_data_keys[0]]['dof'].shape[0], 'frames')
    print("Speed: ", speed)
    print()

    if 'contact_mask' in curr_motion.keys():
        contact_mask = curr_motion['contact_mask']
    else:
        contact_mask = None
    curr_time = 0
    resave = False


    # 使用 atom 机器人的 XML 文件
    humanoid_xml = "./humanoidverse/data/robots/atom/atom.xml"
    print(humanoid_xml)
    
    vis_smpl = False if 'vis_smpl' not in cfg else cfg.vis_smpl
    vis_tau_key = 'tau' if 'vis_tau_key' not in cfg else cfg.vis_tau_key
    vis_tau = vis_tau_key in curr_motion if 'vis_tau' not in cfg else cfg.vis_tau
    vis_contact = 'contact_mask' in curr_motion if 'vis_contact' not in cfg else cfg.vis_contact
    
    if vis_smpl: assert 'smpl_joints' in curr_motion
    if vis_tau: assert vis_tau_key in curr_motion and not vis_contact
    if vis_contact: assert 'contact_mask' in curr_motion and not vis_tau

    if not vis_smpl:
        # 使用 atom 机器人的配置文件
        cfg_robot = OmegaConf.load("humanoidverse/config/robot/atom/atom.yaml")
        humanoid_fk = Humanoid_Batch(cfg_robot['robot']['motion'])  # load forward kinematics model
        pose_aa = torch.from_numpy(curr_motion['pose_aa']).unsqueeze(0)
        root_trans = torch.from_numpy(curr_motion['root_trans_offset']).unsqueeze(0)
        fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)
        joint_gt = fk_return.global_translation_extend[0]
    
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    
    print("Init Pose: ",(np.array(np.concatenate(
        [curr_motion['root_trans_offset'][0],curr_motion['root_rot'][0][[3, 0, 1, 2]], curr_motion['dof'][0]]
        ),dtype=np.float32)).__repr__())
    
    # 在启动viewer之前,设置PKL的第一帧姿态
    mj_data.qpos[:3] = curr_motion['root_trans_offset'][0]
    mj_data.qpos[3:7] = curr_motion['root_rot'][0][[3, 0, 1, 2]]  # xyzw 2 wxyz
    # atom 机器人是 27 DOF
    mj_data.qpos[7:] = curr_motion['dof'][0]
    mujoco.mj_forward(mj_model, mj_data)  # 更新模型状态
    
    # breakpoint()
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        
        viewer.cam.lookat[:] = np.array([0,0,0.7])
        viewer.cam.distance = 3.0        
        viewer.cam.azimuth = 180         
        viewer.cam.elevation = -30                      # 负值表示从上往下看viewer
        
        for _ in range(50):
                # not display the ball
            # add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 0]))
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        
        while viewer.is_running():
            step_start = time.time()
            
            if not paused:
                if rewind:
                    time_step -= 1
                else:
                    time_step += 1
                
                if time_step >= curr_motion['dof'].shape[0]:
                    time_step = 0  # reset
                elif time_step < 0:
                    time_step = curr_motion['dof'].shape[0] - 1
                
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][time_step]
                mj_data.qpos[3:7] = curr_motion['root_rot'][time_step][[3, 0, 1, 2]]  # xyzw 2 wxyz
                mj_data.qpos[7:] = curr_motion['dof'][time_step]
                mujoco.mj_forward(mj_model, mj_data)
                
                # Visualize contact if available
                if vis_contact and contact_mask is not None:
                    # Left foot
                    if contact_mask[time_step, 0] > 0.5:
                        add_visual_capsule(viewer.user_scn, 
                                         mj_data.xpos[mj_model.body('left_ankle_roll_link').id], 
                                         mj_data.xpos[mj_model.body('left_ankle_roll_link').id] + np.array([0, 0, 0.01]), 
                                         0.05, np.array([0, 1, 0, 0.5]))
                    # Right foot
                    if contact_mask[time_step, 1] > 0.5:
                        add_visual_capsule(viewer.user_scn, 
                                         mj_data.xpos[mj_model.body('right_ankle_roll_link').id], 
                                         mj_data.xpos[mj_model.body('right_ankle_roll_link').id] + np.array([0, 0, 0.01]), 
                                         0.05, np.array([0, 1, 0, 0.5]))
                
                # Display frame info
                viewer.opt.label = 1  # Enable label
                
            # update the scene
            viewer.sync()
            
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            time_until_next_step /= speed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

