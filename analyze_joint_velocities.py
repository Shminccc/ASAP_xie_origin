#!/usr/bin/env python3
"""
关节速度分析脚本
分析PKL文件中每个关节的速度曲线和统计信息
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def analyze_joint_velocities(pkl_path, save_plots=False, output_dir="velocity_analysis"):
    """
    分析PKL文件中每个关节的速度
    
    Args:
        pkl_path: PKL文件路径
        save_plots: 是否保存速度曲线图
        output_dir: 输出目录
    """
    
    print(f"📊 分析关节速度: {pkl_path}")
    print("=" * 80)
    
    # 加载PKL文件
    try:
        data = joblib.load(pkl_path)
        print(f"✅ 成功加载文件")
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return
    
    print(f"文件键: {list(data.keys())}")
    
    # 获取第一个运动数据
    key = list(data.keys())[0]
    motion = data[key]
    
    print(f"\n📁 运动: {key}")
    print(f"帧数: {motion['dof'].shape[0]}")
    print(f"DOF数: {motion['dof'].shape[1]}")
    print(f"FPS: {motion['fps']}")
    
    # 计算关节速度 (差分)
    dof_positions = motion['dof']  # (frames, 27)
    dt = 1.0 / motion['fps']  # 时间步长
    dof_velocities = np.diff(dof_positions, axis=0) / dt  # (frames-1, 27)
    
    print(f"\n⏱️  时间信息:")
    print(f"总时长: {motion['dof'].shape[0] / motion['fps']:.2f} 秒")
    print(f"时间步长: {dt:.3f} 秒")
    
    # 关节名称 (按atom.yaml顺序)
    joint_names = [
        'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
        'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
        'waist_yaw',
        'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow_pitch', 'left_elbow_roll', 'left_wrist_pitch', 'left_wrist_yaw',
        'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow_pitch', 'right_elbow_roll', 'right_wrist_pitch', 'right_wrist_yaw'
    ]
    
    # URDF速度限制
    urdf_limits = [
        14, 10, 10, 14, 12, 12,  # 左腿
        14, 10, 10, 14, 12, 12,  # 右腿
        52,  # 腰
        2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62,  # 左臂
        2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62   # 右臂
    ]
    
    print(f"\n📈 关节速度统计 (rad/s):")
    print("-" * 100)
    print(f"{'关节名称':<20} {'最大速度':<10} {'最小速度':<10} {'平均速度':<10} {'标准差':<10} {'URDF限制':<10} {'超限比例':<10}")
    print("-" * 100)
    
    max_velocities = []
    exceeded_joints = []
    
    for i, name in enumerate(joint_names):
        vel = dof_velocities[:, i]
        max_vel = np.max(np.abs(vel))  # 最大绝对值速度
        min_vel = np.min(vel)
        mean_vel = np.mean(np.abs(vel))  # 平均绝对值速度
        std_vel = np.std(vel)
        urdf_limit = urdf_limits[i]
        ratio = max_vel / urdf_limit
        
        max_velocities.append(max_vel)
        
        if ratio > 1.0:
            exceeded_joints.append((name, max_vel, urdf_limit, ratio))
            status = "⚠️"
        else:
            status = "✅"
            
        print(f"{name:<20} {max_vel:<10.3f} {min_vel:<10.3f} {mean_vel:<10.3f} {std_vel:<10.3f} {urdf_limit:<10.1f} {ratio:<10.2f} {status}")
    
    print("-" * 100)
    
    # 按关节组分析
    print(f"\n🎯 关节组分析:")
    leg_indices = list(range(12))  # 0-11
    waist_indices = [12]  # 12
    arm_indices = list(range(13, 27))  # 13-26
    
    leg_max_vel = max([max_velocities[i] for i in leg_indices])
    waist_max_vel = max_velocities[waist_indices[0]]
    arm_max_vel = max([max_velocities[i] for i in arm_indices])
    
    print(f"🦵 腿部最大速度: {leg_max_vel:.3f} rad/s ({leg_max_vel*180/np.pi:.1f}°/s)")
    print(f"🔄 腰部最大速度: {waist_max_vel:.3f} rad/s ({waist_max_vel*180/np.pi:.1f}°/s)")
    print(f"🤲 手臂最大速度: {arm_max_vel:.3f} rad/s ({arm_max_vel*180/np.pi:.1f}°/s)")
    
    # 超限总结
    if exceeded_joints:
        print(f"\n🚨 发现 {len(exceeded_joints)} 个关节超限!")
        for name, actual, limit, ratio in exceeded_joints:
            print(f"  {name}: {actual:.3f} rad/s > {limit:.1f} rad/s (超限 {ratio:.1f}倍)")
    else:
        print(f"\n✅ 所有关节都在URDF限制范围内!")
    
    # 保存详细数据
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存速度数据为CSV
        csv_path = os.path.join(output_dir, "joint_velocities.csv")
        with open(csv_path, 'w') as f:
            f.write("joint_name,max_velocity,min_velocity,mean_velocity,std_velocity,urdf_limit,exceed_ratio\n")
            for i, name in enumerate(joint_names):
                vel = dof_velocities[:, i]
                max_vel = np.max(np.abs(vel))
                min_vel = np.min(vel)
                mean_vel = np.mean(np.abs(vel))
                std_vel = np.std(vel)
                urdf_limit = urdf_limits[i]
                ratio = max_vel / urdf_limit
                f.write(f"{name},{max_vel:.6f},{min_vel:.6f},{mean_vel:.6f},{std_vel:.6f},{urdf_limit:.1f},{ratio:.6f}\n")
        print(f"\n💾 速度数据已保存到: {csv_path}")
        
        # 绘制速度曲线图
        time_axis = np.arange(dof_velocities.shape[0]) * dt
        
        # 按关节组绘制
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 腿部关节
        axes[0].set_title('腿部关节速度 (rad/s)', fontsize=14, fontweight='bold')
        for i in leg_indices:
            axes[0].plot(time_axis, dof_velocities[:, i], label=joint_names[i], alpha=0.7)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel('速度 (rad/s)')
        
        # 腰部关节
        axes[1].set_title('腰部关节速度 (rad/s)', fontsize=14, fontweight='bold')
        axes[1].plot(time_axis, dof_velocities[:, waist_indices[0]], label=joint_names[waist_indices[0]], color='red', linewidth=2)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('速度 (rad/s)')
        
        # 手臂关节
        axes[2].set_title('手臂关节速度 (rad/s)', fontsize=14, fontweight='bold')
        for i in arm_indices:
            axes[2].plot(time_axis, dof_velocities[:, i], label=joint_names[i], alpha=0.7)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylabel('速度 (rad/s)')
        axes[2].set_xlabel('时间 (s)')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "joint_velocities.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 速度曲线图已保存到: {plot_path}")
        
        # 绘制最大速度对比图
        plt.figure(figsize=(15, 8))
        x_pos = np.arange(len(joint_names))
        
        plt.bar(x_pos, max_velocities, alpha=0.7, label='实际最大速度')
        plt.bar(x_pos, urdf_limits, alpha=0.3, label='URDF限制', color='red')
        
        plt.xlabel('关节')
        plt.ylabel('速度 (rad/s)')
        plt.title('关节最大速度 vs URDF限制')
        plt.xticks(x_pos, joint_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 标记超限关节
        for i, (name, actual, limit, ratio) in enumerate(exceeded_joints):
            joint_idx = joint_names.index(name)
            plt.annotate(f'{ratio:.1f}x', 
                        xy=(joint_idx, actual), 
                        xytext=(joint_idx, actual + 0.5),
                        ha='center', va='bottom',
                        fontweight='bold', color='red')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "velocity_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"📊 速度对比图已保存到: {comparison_path}")
        
        print(f"\n📁 所有输出文件保存在: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='分析PKL文件中关节速度')
    parser.add_argument('pkl_path', help='PKL文件路径')
    parser.add_argument('--save-plots', action='store_true', help='保存速度曲线图')
    parser.add_argument('--output-dir', default='velocity_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        print(f"❌ 文件不存在: {args.pkl_path}")
        return
    
    analyze_joint_velocities(args.pkl_path, args.save_plots, args.output_dir)

if __name__ == "__main__":
    main()
