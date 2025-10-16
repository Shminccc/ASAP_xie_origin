#!/usr/bin/env python3
"""
快速关节速度统计脚本
只输出统计信息，不生成图表
"""

import joblib
import numpy as np
import argparse
import os

def quick_velocity_stats(pkl_path):
    """
    快速统计PKL文件中每个关节的速度
    """
    
    print(f"📊 快速速度统计: {os.path.basename(pkl_path)}")
    print("=" * 80)
    
    # 加载PKL文件
    try:
        data = joblib.load(pkl_path)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return
    
    # 获取第一个运动数据
    key = list(data.keys())[0]
    motion = data[key]
    
    print(f"运动: {key}")
    print(f"帧数: {motion['dof'].shape[0]}, FPS: {motion['fps']}, 时长: {motion['dof'].shape[0] / motion['fps']:.2f}s")
    
    # 计算关节速度
    dof_positions = motion['dof']
    dt = 1.0 / motion['fps']
    dof_velocities = np.diff(dof_positions, axis=0) / dt
    
    # 关节名称
    joint_names = [
        'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
        'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
        'waist_yaw',
        'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow_pitch', 'left_elbow_roll', 'left_wrist_pitch', 'left_wrist_yaw',
        'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow_pitch', 'right_elbow_roll', 'right_wrist_pitch', 'right_wrist_yaw'
    ]
    
    # URDF限制
    urdf_limits = [
        14, 10, 10, 14, 12, 12,  # 左腿
        14, 10, 10, 14, 12, 12,  # 右腿
        52,  # 腰
        2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62,  # 左臂
        2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62   # 右臂
    ]
    
    print(f"\n📈 关节最大速度 (rad/s):")
    print("-" * 80)
    print(f"{'关节名称':<20} {'最大速度':<10} {'URDF限制':<10} {'超限比例':<10} {'状态':<5}")
    print("-" * 80)
    
    exceeded_count = 0
    max_overall = 0
    
    for i, name in enumerate(joint_names):
        vel = dof_velocities[:, i]
        max_vel = np.max(np.abs(vel))
        urdf_limit = urdf_limits[i]
        ratio = max_vel / urdf_limit
        
        if max_vel > max_overall:
            max_overall = max_vel
            
        if ratio > 1.0:
            exceeded_count += 1
            status = "⚠️"
        else:
            status = "✅"
            
        print(f"{name:<20} {max_vel:<10.3f} {urdf_limit:<10.1f} {ratio:<10.2f} {status:<5}")
    
    print("-" * 80)
    print(f"\n🎯 总结:")
    print(f"  整体最大速度: {max_overall:.3f} rad/s ({max_overall*180/np.pi:.1f}°/s)")
    print(f"  超限关节数量: {exceeded_count}/27")
    
    if exceeded_count > 0:
        print(f"  ⚠️  发现 {exceeded_count} 个关节超限!")
    else:
        print(f"  ✅ 所有关节都在限制范围内!")

def main():
    parser = argparse.ArgumentParser(description='快速统计PKL文件中关节速度')
    parser.add_argument('pkl_path', help='PKL文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        print(f"❌ 文件不存在: {args.pkl_path}")
        return
    
    quick_velocity_stats(args.pkl_path)

if __name__ == "__main__":
    main()
