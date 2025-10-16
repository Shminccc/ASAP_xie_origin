#!/usr/bin/env python3
"""
ATOM 机器人动作可视化工具 - 简化版本
不依赖完整的 Hydra 配置系统

使用方法：
    python robot_motion_process/vis_q_mj_atom_simple.py <pkl_file> [speed]
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import joblib

# 全局变量
time_step = 0
paused = False
rewind = False
speed = 1.0
dt = 1 / 60.0


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        point1.astype(np.float64), point2.astype(np.float64))


def key_callback(keycode):
    """键盘回调函数"""
    global time_step, paused, rewind, speed
    
    if chr(keycode) == "R":
        print("重置")
        time_step = 0
    elif chr(keycode) == " ":
        print("暂停/播放")
        paused = not paused
    elif keycode == 256 or chr(keycode) == "Q":
        print("退出")
        sys.exit()
    elif chr(keycode) == "L":
        print("加速")
        speed *= 1.5
        print(f"当前速度: {speed}x")
    elif chr(keycode) == "K":
        print("减速")
        speed /= 1.5
        print(f"当前速度: {speed}x")
    elif chr(keycode) == "J":
        print("倒放")
        rewind = not rewind
    elif keycode == 263:  # 左箭头
        print("上一帧")
        time_step -= 1
        paused = True
    elif keycode == 262:  # 右箭头
        print("下一帧")
        time_step += 1
        paused = True


def main():
    global time_step, paused, rewind, speed
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python vis_q_mj_atom_simple.py <pkl_file> [speed]")
        print("示例: python vis_q_mj_atom_simple.py humanoidverse/data/motions/atom/Walking_3_poses.pkl 1.0")
        sys.exit(1)
    
    motion_file = sys.argv[1]
    if len(sys.argv) > 2:
        speed = float(sys.argv[2])
    
    print(f"\n🤖 ATOM 机器人动作可视化")
    print("=" * 60)
    print(f"📁 运动文件: {motion_file}")
    print(f"⚡ 播放速度: {speed}x")
    print()
    print("🎮 控制说明:")
    print("  空格键    - 暂停/播放")
    print("  R键       - 重置到开始")
    print("  L键       - 加速播放 (1.5x)")
    print("  K键       - 减速播放 (/1.5)")
    print("  J键       - 切换倒放")
    print("  左箭头    - 上一帧")
    print("  右箭头    - 下一帧")
    print("  Q键       - 退出")
    print()
    print("🎨 接触可视化:")
    print("  🔴 红色大球  - 左脚接触地面")
    print("  🔵 蓝色大球  - 右脚接触地面")
    print("  🟢 绿色小球  - 左脚在空中")
    print("  🟡 黄色小球  - 右脚在空中")
    print("=" * 60)
    print()
    
    # 加载运动数据
    print("📦 加载运动数据...")
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    curr_motion = motion_data[motion_data_keys[0]]
    
    num_frames = curr_motion['dof'].shape[0]
    fps = curr_motion.get('fps', 30)
    duration = num_frames / fps
    
    print(f"   帧数: {num_frames}")
    print(f"   FPS: {fps}")
    print(f"   时长: {duration:.2f}秒")
    print(f"   DOF: {curr_motion['dof'].shape[1]}")
    print()
    
    # 检查是否有接触遮罩
    contact_mask = curr_motion.get('contact_mask', None)
    if contact_mask is not None:
        print(f"   接触遮罩: 是 (形状: {contact_mask.shape})")
    else:
        print(f"   接触遮罩: 否")
    print()
    
    # 加载机器人模型
    humanoid_xml = "./humanoidverse/data/robots/atom/atom.xml"
    print(f"🤖 加载 ATOM 机器人模型: {humanoid_xml}")
    
    if not os.path.exists(humanoid_xml):
        print(f"\n❌ 错误: 找不到机器人模型文件: {humanoid_xml}")
        print("请确保你在正确的目录中运行此脚本")
        sys.exit(1)
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    
    print(f"   机器人 DOF: {mj_model.nq - 7}")  # 减去 7 个 free joint 的自由度
    print()
    
    # 设置初始姿态
    mj_data.qpos[:3] = curr_motion['root_trans_offset'][0]
    mj_data.qpos[3:7] = curr_motion['root_rot'][0][[3, 0, 1, 2]]  # xyzw -> wxyz
    mj_data.qpos[7:] = curr_motion['dof'][0]
    mujoco.mj_forward(mj_model, mj_data)
    
    print("✅ 启动可视化窗口...")
    print()
    
    # 启动 viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        # 设置相机
        viewer.cam.lookat[:] = np.array([0, 0, 0.7])
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30
        
        # 主循环
        while viewer.is_running():
            step_start = time.time()
            
            if not paused:
                # 更新帧
                if rewind:
                    time_step -= 1
                else:
                    time_step += 1
                
                # 循环播放
                if time_step >= num_frames:
                    time_step = 0
                elif time_step < 0:
                    time_step = num_frames - 1
                
                # 更新机器人姿态
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][time_step]
                mj_data.qpos[3:7] = curr_motion['root_rot'][time_step][[3, 0, 1, 2]]
                mj_data.qpos[7:] = curr_motion['dof'][time_step]
                mujoco.mj_forward(mj_model, mj_data)
            
            # 清空之前的可视化（避免累积）
            viewer.user_scn.ngeom = 0
            
            # 可视化接触点（始终显示当前帧的状态）
            if contact_mask is not None:
                # 左脚 - 接触时显示红色，不接触时显示淡绿色
                left_foot_pos = mj_data.xpos[mj_model.body('left_ankle_roll_link').id]
                if contact_mask[time_step, 0] > 0.5:
                    # 接触 = 红色大球
                    add_visual_capsule(viewer.user_scn,
                                     left_foot_pos,
                                     left_foot_pos + np.array([0, 0, 0.02]),
                                     0.06, np.array([1, 0, 0, 0.9]))  # 红色，更大更明显
                else:
                    # 不接触 = 淡绿色小球（空中）
                    add_visual_capsule(viewer.user_scn,
                                     left_foot_pos,
                                     left_foot_pos + np.array([0, 0, 0.01]),
                                     0.03, np.array([0, 1, 0, 0.3]))  # 淡绿色
                
                # 右脚 - 接触时显示蓝色，不接触时显示淡黄色
                right_foot_pos = mj_data.xpos[mj_model.body('right_ankle_roll_link').id]
                if contact_mask[time_step, 1] > 0.5:
                    # 接触 = 蓝色大球
                    add_visual_capsule(viewer.user_scn,
                                     right_foot_pos,
                                     right_foot_pos + np.array([0, 0, 0.02]),
                                     0.06, np.array([0, 0, 1, 0.9]))  # 蓝色，更大更明显
                else:
                    # 不接触 = 淡黄色小球（空中）
                    add_visual_capsule(viewer.user_scn,
                                     right_foot_pos,
                                     right_foot_pos + np.array([0, 0, 0.01]),
                                     0.03, np.array([1, 1, 0, 0.3]))  # 淡黄色
            
            # 同步 viewer
            viewer.sync()
            
            # 时间控制
            time_until_next_step = dt - (time.time() - step_start)
            time_until_next_step /= speed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

