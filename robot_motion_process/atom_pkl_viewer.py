#!/usr/bin/env python3
"""
ATOM 机器人动作可视化工具 - 交互式版本
自动列出所有可用的pkl文件并让用户选择

使用方法：
    python robot_motion_process/atom_pkl_viewer.py
"""

import os
import subprocess
import sys


def get_available_motions():
    """获取所有可用的atom动作文件，优先显示带contact_mask的文件"""
    motion_files = []
    motion_dirs = [
        "humanoidverse/data/motions/atom_contact_mask",  # 优先显示处理后的文件
        "humanoidverse/data/motions/atom",
        "example/motion_data",
    ]

    for motion_dir in motion_dirs:
        if os.path.exists(motion_dir):
            for file in os.listdir(motion_dir):
                if file.endswith('.pkl'):
                    full_path = os.path.join(motion_dir, file)
                    # 避免重复添加同名文件
                    if full_path not in motion_files:
                        motion_files.append(full_path)

    return sorted(motion_files)


def show_motion_menu(motion_files):
    """显示动作选择菜单"""
    print("\n🤖 ATOM 机器人动作可视化工具")
    print("=" * 60)
    print("📁 可用的动作文件:")
    print()

    for i, file in enumerate(motion_files, 1):
        filename = os.path.basename(file)
        # 移除.pkl扩展名并美化显示
        display_name = filename.replace('.pkl', '').replace('_', ' ').title()
        folder = os.path.basename(os.path.dirname(file))
        print(f"  {i:2d}. {display_name:30s} [{folder}]")

    print(f"  {len(motion_files) + 1:2d}. 🚪 退出")
    print()


def run_visualization(pkl_file, speed=1.0):
    """运行可视化工具"""
    print(f"\n🚀 启动 ATOM 机器人可视化: {os.path.basename(pkl_file)}")
    print(f"⚡ 播放速度: {speed}x")
    print()

    print("🎮 控制说明:")
    print("  空格键    - 暂停/播放")
    print("  R键       - 重置到开始")
    print("  L键       - 加速播放")
    print("  K键       - 减速播放")
    print("  J键       - 切换倒放")
    print("  左右箭头  - 逐帧控制")
    print("  Q键       - 退出可视化")
    print("=" * 60)

    cmd = [
        "python",
        "robot_motion_process/vis_q_mj_atom_simple.py",
        pkl_file,
        str(speed)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查:")
        print("  1. 确保运动文件格式正确")
        print("  2. 确保 ATOM 机器人模型文件存在")
        print("  3. 确保运动数据是 27 DOF 的 ATOM 机器人数据")
    except KeyboardInterrupt:
        print("\n👋 用户中断，返回主菜单")
    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}")


def get_speed_setting():
    """获取播放速度设置"""
    while True:
        try:
            speed_input = input("⚡ 请输入播放速度 (默认1.0): ").strip()
            if not speed_input:
                return 1.0
            speed = float(speed_input)
            if speed > 0:
                return speed
            else:
                print("❌ 速度必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🤖 ATOM 机器人 (27 DOF) 动作可视化工具")
    print("=" * 60)
    
    motion_files = get_available_motions()

    if not motion_files:
        print("\n❌ 未找到 ATOM 机器人的 pkl 动作文件")
        print("请确保以下目录中包含 .pkl 文件:")
        print("  - humanoidverse/data/motions/atom/")
        print("  - example/motion_data/")
        return

    while True:
        show_motion_menu(motion_files)

        try:
            choice = input("🔢 请选择要查看的动作 (输入数字): ").strip()

            if not choice:
                continue

            choice_num = int(choice)

            if choice_num == len(motion_files) + 1:
                print("\n👋 再见！")
                break
            elif 1 <= choice_num <= len(motion_files):
                selected_file = motion_files[choice_num - 1]

                # 询问播放速度
                speed = get_speed_setting()

                # 运行可视化
                run_visualization(selected_file, speed)

                # 询问是否继续
                while True:
                    continue_choice = input("\n🔄 是否继续选择其他动作? (y/n): ").strip().lower()
                    if continue_choice in ['y', 'yes', '是']:
                        break
                    elif continue_choice in ['n', 'no', '否']:
                        print("\n👋 再见！")
                        return
                    else:
                        print("❌ 请输入 y 或 n")
            else:
                print(f"❌ 请输入 1 到 {len(motion_files) + 1} 之间的数字")

        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()

