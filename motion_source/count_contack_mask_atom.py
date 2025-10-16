import glob
import os
import sys
import os.path as osp
import numpy as np

sys.path.append(os.getcwd())

from utils.torch_humanoid_batch import Humanoid_Batch
import torch
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from scipy.spatial.transform import Rotation as sRot

def foot_detect(positions, fid_l, fid_r, thres=0.002):
    """
    检测脚部接触（ATOM 适配版）
    
    Args:
        positions: (T, N, 3) 全局关节位置
        fid_l: 左脚在 body_names 中的索引
        fid_r: 右脚在 body_names 中的索引
        thres: 速度阈值（m^2）
    """
    positions = positions.numpy() if isinstance(positions, torch.Tensor) else positions
    velfactor, heightfactor = np.array([thres]), np.array([0.12]) 
    
    # 左脚接触检测
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:, fid_l, 2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & 
              (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.expand_dims(feet_l, axis=1)
    feet_l = np.concatenate([np.array([[1.]], dtype=np.float32), feet_l], axis=0)

    # 右脚接触检测
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:, fid_r, 2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & 
              (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.expand_dims(feet_r, axis=1)
    feet_r = np.concatenate([np.array([[1.]], dtype=np.float32), feet_r], axis=0)
    
    return feet_l, feet_r

def process_motion_atom(motion, atom_body_names, atom_xml_path):
    """
    处理 ATOM 运动数据，计算接触掩码
    
    Args:
        motion: 运动数据字典
        atom_body_names: ATOM 机器人的 body 名称列表
        atom_xml_path: ATOM 机器人 XML 文件的完整路径
    
    Returns:
        添加了 contact_mask 和 smpl_joints 的运动数据
    """
    device = torch.device("cpu")
    
    # ATOM 的 pkl 文件通常已包含 pose_aa，直接使用
    if 'pose_aa' not in motion.keys():
        raise ValueError(
            "ATOM motion data must contain 'pose_aa'. "
            "Please ensure the motion file is properly formatted for ATOM robot."
        )
    
    # 构建临时配置用于 FK
    tmp_cfg = OmegaConf.create({
        'robot': {
            'motion': {
                'body_names': atom_body_names,
                'asset': {
                    'assetFileName': atom_xml_path,  # 使用完整路径
                    'assetRoot': ''  # 空字符串，因为我们已经提供了完整路径
                }
            }
        }
    })
    
    humanoid_fk = Humanoid_Batch(tmp_cfg.robot)
    
    pose_aa = torch.from_numpy(motion['pose_aa']).unsqueeze(0)
    root_trans = torch.from_numpy(motion['root_trans_offset']).unsqueeze(0)
    
    fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)
    
    # 获取全局关节位置
    if hasattr(fk_return, 'global_translation_extend'):
        global_translation = fk_return.global_translation_extend[0]
    else:
        global_translation = fk_return.global_translation[0]
    
    # 查找 ATOM 的脚部索引
    try:
        fid_l = atom_body_names.index("left_ankle_roll_link")
        fid_r = atom_body_names.index("right_ankle_roll_link")
    except ValueError as e:
        raise ValueError(
            f"Cannot find ankle links in body_names. "
            f"Expected 'left_ankle_roll_link' and 'right_ankle_roll_link'. "
            f"Available names: {atom_body_names}"
        ) from e
    
    print(f"  使用脚部索引 - 左脚: {fid_l}, 右脚: {fid_r}")
    
    # 检测接触
    feet_l, feet_r = foot_detect(global_translation, fid_l, fid_r)
    
    motion['contact_mask'] = np.concatenate([feet_l, feet_r], axis=-1)
    motion['smpl_joints'] = global_translation.detach().numpy()
    
    return motion


def main():
    """
    批量处理 ATOM 运动数据，添加接触掩码
    
    用法:
        cd /home/dobot/Desktop/extracted_files
        python motion_source/count_contack_mask_atom.py /path/to/atom/motions
    """
    import argparse
    parser = argparse.ArgumentParser(description='为 ATOM 运动数据添加接触掩码')
    parser.add_argument('input_folder', type=str, help='包含 ATOM .pkl 文件的输入文件夹')
    parser.add_argument('--config', type=str, 
                       default='humanoidverse/config/robot/atom/atom.yaml',
                       help='ATOM 配置文件路径')
    args = parser.parse_args()
    
    folder_path = args.input_folder
    
    # 生成输出文件夹名
    if folder_path[-1] == '/':
        target_folder_path = folder_path[:-1] + '_contact_mask'
    else:
        target_folder_path = folder_path + '_contact_mask'
    
    os.makedirs(target_folder_path, exist_ok=True)
    print(f"📂 输入文件夹: {folder_path}")
    print(f"📂 输出文件夹: {target_folder_path}")
    
    # 加载 ATOM 配置
    print(f"📄 加载配置文件: {args.config}")
    cfg = OmegaConf.load(args.config)
    atom_body_names = cfg.robot.motion.body_names
    
    # 构建 ATOM XML 文件的绝对路径
    atom_xml_path = os.path.join(
        cfg.robot.motion.asset.assetRoot,
        cfg.robot.motion.asset.assetFileName
    )
    if not os.path.isabs(atom_xml_path):
        atom_xml_path = os.path.join(os.getcwd(), atom_xml_path)
    
    if not os.path.exists(atom_xml_path):
        print(f"❌ 错误: 找不到 ATOM XML 文件: {atom_xml_path}")
        return
    
    print(f"✅ 加载 ATOM 配置，共 {len(atom_body_names)} 个 bodies")
    print(f"✅ XML 文件: {atom_xml_path}")
    
    # 获取已处理的文件列表
    target_folder_list = os.listdir(target_folder_path) if os.path.exists(target_folder_path) else []
    
    # 遍历输入文件夹中的所有 pkl 文件
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    print(f"\n找到 {len(pkl_files)} 个 .pkl 文件")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for filename in pkl_files:
        output_filename = filename.replace('.pkl', '_cont_mask.pkl')
        
        # 跳过已处理的文件
        if output_filename in target_folder_list:
            print(f"⏭️  跳过 {filename} (已存在)")
            skipped_count += 1
            continue
        
        motion_file = os.path.join(folder_path, filename)
        print(f"\n🔧 处理: {filename}")
        
        try:
            # 加载运动数据
            motion_data = joblib.load(motion_file)
            motion_data_keys = list(motion_data.keys())
            
            if len(motion_data_keys) == 0:
                print(f"  ⚠️  警告: 文件为空")
                error_count += 1
                continue
            
            print(f"  包含 {len(motion_data_keys)} 个运动序列")
            
            # 处理第一个运动序列
            motion = process_motion_atom(motion_data[motion_data_keys[0]], atom_body_names, atom_xml_path)
            
            # 保存结果
            save_data = {motion_data_keys[0]: motion}
            dumped_file = os.path.join(target_folder_path, output_filename)
            joblib.dump(save_data, dumped_file)
            
            # 统计信息
            contact_left = motion['contact_mask'][:, 0].sum()
            contact_right = motion['contact_mask'][:, 1].sum()
            total_frames = len(motion['contact_mask'])
            
            print(f"  ✅ 成功! 帧数: {total_frames}")
            print(f"     左脚接触: {int(contact_left)}/{total_frames} ({contact_left/total_frames*100:.1f}%)")
            print(f"     右脚接触: {int(contact_right)}/{total_frames} ({contact_right/total_frames*100:.1f}%)")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            error_count += 1
            continue
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"📊 处理完成!")
    print(f"  成功: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    print(f"  失败: {error_count} 个文件")
    print(f"  输出目录: {target_folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
