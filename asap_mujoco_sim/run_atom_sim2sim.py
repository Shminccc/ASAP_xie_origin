#!/usr/bin/env python3
"""
ATOM Robot Sim2Sim Trajectory Collection
基于 mujoco_track_with_processing.py 适配到 ATOM 机器人
"""
import sys
import os

# 设置工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 导入原始脚本的函数
from mujoco_track_with_processing import (
    read_conf,
    run_and_save_mujoco,
    process_motion_data
)
from datetime import datetime


def main():
    """
    ATOM 机器人轨迹采集主函数
    """
    # 读取 ATOM 配置
    config_file = os.path.join(current_dir, "atom_config", "mujoco_config_atom.yaml")
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请先确保 atom_config/mujoco_config_atom.yaml 已正确配置")
        return
    
    print(f"📋 加载配置文件: {config_file}")
    cfg = read_conf(config_file)
    
    # 检查必要文件
    if not os.path.exists(cfg.xml_path):
        print(f"❌ XML 文件不存在: {cfg.xml_path}")
        print("请确保 atom_urdf/atom.xml 文件存在")
        return
    
    if not os.path.exists(cfg.policy_path):
        print(f"❌ 策略文件不存在: {cfg.policy_path}")
        print("请在配置文件中设置正确的 policy_path")
        return
    
    # 验证配置
    assert cfg.num_actions == 27, f"ATOM 应有 27 DOF，当前配置为 {cfg.num_actions}"
    assert len(cfg.kps) == 27, f"kps 应有 27 个值，当前有 {len(cfg.kps)}"
    assert len(cfg.kds) == 27, f"kds 应有 27 个值，当前有 {len(cfg.kds)}"
    
    print("✅ 配置验证通过")
    print(f"  - DOF: {cfg.num_actions}")
    print(f"  - Episode 步数: {cfg.episode_steps}")
    print(f"  - 总步数: {cfg.total_steps}")
    print(f"  - 控制频率: {1.0 / (cfg.simulation_dt * cfg.control_decimation):.1f} Hz")
    print(f"  - Termination: {'启用' if cfg.use_termination else '禁用'}")
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_save_path = os.path.join(current_dir, f"{timestamp}_atom_motion_raw.pkl")
    
    print(f"\n=== 开始 ATOM 轨迹采集 ===")
    print(f"原始数据将保存到: {raw_save_path}")
    
    # 步骤1：运行仿真并保存原始数据
    saved_path = run_and_save_mujoco(cfg, raw_save_path)
    
    if saved_path and cfg.auto_process:
        print(f"\n=== 开始自动处理数据 ===")
        # 步骤2：自动处理数据
        processed_path = process_motion_data(saved_path, cfg)
        
        if processed_path:
            print(f"\n🎉 ATOM 轨迹采集完成!")
            print(f"📁 原始数据: {saved_path}")
            print(f"📁 处理后数据: {processed_path}")
            print(f"\n💡 处理后的数据可直接用于训练，格式已匹配 humanoidverse")
        else:
            print(f"\n⚠️ 数据处理失败，原始数据保留在: {saved_path}")
    else:
        print(f"\n✅ 轨迹采集完成，数据保存在: {saved_path}")
        if not cfg.auto_process:
            print("💡 如需处理数据，请在配置文件中设置 auto_process: true")
    
    print("\n" + "="*50)
    print("✅ 任务完成")
    print("="*50)


if __name__ == '__main__':
    main()

