# ATOM 脚踝 Delta Action 训练适配总结

## 修改文件
`humanoidverse/agents/delta_a/train_delta_a_ankle_atom.py`

## 关键修改点

### 1️⃣ 脚踝关节索引 (保持不变)

```python
# ATOM 脚踝关节索引 (ATOM 是 27 DOF)
# 关节顺序：左腿6 + 右腿6 + 腰1 + 左臂7 + 右臂7
# left_ankle_pitch_joint: 4, left_ankle_roll_joint: 5
# right_ankle_pitch_joint: 10, right_ankle_roll_joint: 11
self.ankle_indices = [4, 5, 10, 11]  # ATOM 和 G1 的脚踝索引恰好相同！
```

**分析**：
- G1 (23 DOF): 左腿6 + 右腿6 + 腰3 + 左臂4 + 右臂4
- ATOM (27 DOF): 左腿6 + 右腿6 + 腰1 + 左臂7 + 右臂7

由于两个机器人的腿部关节顺序完全一致，脚踝关节的索引恰好相同：
- 左脚踝 pitch: 索引 4
- 左脚踝 roll: 索引 5  
- 右脚踝 pitch: 索引 10
- 右脚踝 roll: 索引 11

### 2️⃣ 动作维度扩展

```python
def _expand_ankle_to_full(self, ankle_actions_4d):
    """将4维脚踝动作扩展为27维全身动作 (ATOM)"""
    batch_size = ankle_actions_4d.shape[0]
    full_actions = torch.zeros(batch_size, 27, device=self.device)  # ✅ 改为 27
    full_actions[:, self.ankle_indices] = ankle_actions_4d
    return full_actions
```

**修改**：
- G1: `torch.zeros(batch_size, 23, device=self.device)`
- ATOM: `torch.zeros(batch_size, 27, device=self.device)`

### 3️⃣ 注释更新

在以下位置更新了注释，明确标注 ATOM 27 DOF：

1. `_rollout_step` 方法中：
   ```python
   # 关键修改：网络输出4维，扩展为27维 (ATOM)
   ankle_delta_4d = policy_state_dict["actions"]  # 4维输出
   delta_actions = self._expand_ankle_to_full(ankle_delta_4d)  # 扩展为27维
   ```

2. `_pre_eval_env_step` 方法中：
   ```python
   # 扩展为27维 (ATOM) 并与pkl相加
   delta_actions = self._expand_ankle_to_full(ankle_actions_4d)
   ```

## 工作原理

### 训练流程

1. **网络输出**：策略网络输出 4 维脚踝残差动作
   ```
   ankle_delta_4d.shape = (batch_size, 4)
   ```

2. **扩展为全身动作**：将 4 维扩展为 27 维，其他关节置零
   ```python
   full_actions = torch.zeros(batch_size, 27)
   full_actions[:, [4, 5, 10, 11]] = ankle_delta_4d
   # 其他 23 个关节的 delta 为 0
   ```

3. **与 PKL 参考动作相加**：
   ```python
   final_actions = delta_actions + pkl_actions
   # delta_actions: (batch, 27) - 只有脚踝4个关节非零
   # pkl_actions: (batch, 27) - 完整的参考动作
   # final_actions: (batch, 27) - 最终执行的动作
   ```

4. **效果**：
   - 脚踝关节 = PKL参考 + 网络残差修正
   - 其他关节 = PKL参考（完全跟随）

### 评估流程

与训练流程相同，使用 `_pre_eval_env_step` 方法处理评估时的动作。

## ATOM vs G1 关节对比

| 机器人 | 总DOF | 腿部 | 腰部 | 手臂 | 脚踝索引 |
|--------|-------|------|------|------|----------|
| G1     | 23    | 12   | 3    | 8    | [4,5,10,11] |
| ATOM   | 27    | 12   | 1    | 14   | [4,5,10,11] |

**关键发现**：由于腿部关节顺序相同，脚踝索引保持一致！

## 验证清单

使用前请确认：

- [ ] 环境配置使用 ATOM robot config
- [ ] `robot.actions_dim = 27`
- [ ] `robot.dof_obs_size = 27`
- [ ] PKL motion file 是 ATOM 格式（27 DOF）
- [ ] 网络输出维度设置为 4（在 `__init__` 中自动处理）

## 使用示例

```bash
python humanoidverse/train_agent.py \
    +simulator=isaacgym \
    +exp=motion_tracking \
    +terrain=terrain_locomotion_plane \
    +robot=atom/atom \
    +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history_atom \
    +rewards=motion_tracking/main \
    robot.motion.motion_file="path/to/atom_motion.pkl" \
    algo.config._target_=humanoidverse.agents.delta_a.train_delta_a_ankle_atom.PPODeltaAAnkleOnly \
    num_envs=2048
```

## 注意事项

1. **维度匹配**：确保所有涉及 DOF 的地方都使用 27 而不是 23
2. **索引一致性**：脚踝索引 [4,5,10,11] 对 ATOM 和 G1 都适用
3. **PKL 格式**：motion file 必须是 ATOM 27 DOF 格式
4. **网络架构**：输入观测维度会自动适配，输出固定为 4 维脚踝动作

## 补充说明

### 为什么脚踝索引相同？

```
G1 (23 DOF):
[0-5]   左腿6个关节 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
[6-11]  右腿6个关节
[12-14] 腰部3个关节
[15-22] 手臂8个关节

ATOM (27 DOF):
[0-5]   左腿6个关节 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)  ✅ 相同
[6-11]  右腿6个关节  ✅ 相同
[12]    腰部1个关节
[13-26] 手臂14个关节

脚踝在两个机器人中的位置：
- 左脚踝: [4, 5]  ✅ 完全相同
- 右脚踝: [10, 11]  ✅ 完全相同
```

这个巧合使得代码迁移变得非常简单！
