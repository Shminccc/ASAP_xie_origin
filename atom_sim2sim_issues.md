# ATOM Sim2Sim 问题报告

## 🚨 **发现的关键问题**

### 1️⃣ **PD Gains 配置格式错误** ⚠️⚠️⚠️
**问题**: 训练配置中的PD gains是字典形式（按关节名），而sim2sim中是列表形式（按顺序）

**训练配置 (atom.yaml)**:
```yaml
stiffness: {
  hip_yaw: 144.89,
  hip_roll: 278.81,
  hip_pitch: 430.24,
  knee: 320.8,
  ankle_pitch: 60.64,
  ankle_roll: 60.64,
  waist_yaw: 443.71,
  shoulder_pitch: 80,
  shoulder_roll: 80,
  shoulder_yaw: 80,
  wrist_pitch: 80,
  wrist_yaw: 80,
  elbow_pitch: 60,
  elbow_roll: 60
}
```

**Sim2Sim 配置 (mujoco_config_atom.yaml)**:
```yaml
kps: [
    # 左腿 6: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    430.24, 278.81, 144.89, 320.8, 60.64, 60.64,
    # 右腿 6
    430.24, 278.81, 144.89, 320.8, 60.64, 60.64,
    # 腰 1: waist_yaw
    443.71,
    # 左臂 7
    80, 80, 80, 60, 60, 80, 80,
    # 右臂 7
    80, 80, 80, 60, 60, 80, 80
]
```

**分析**: 需要确认关节顺序是否正确！

---

### 2️⃣ **观测缩放系数不一致** ⚠️
**训练配置 (deepmimic_a2c_nolinvel_LARGEnoise_history_atom.yaml)**:
```yaml
obs_scales: {
  base_ang_vel: 0.25,      # ✅ 一致
  dof_pos: 1.0,            # ✅ 一致
  dof_vel: 0.05,           # ✅ 一致
  projected_gravity: 1.0,  # ✅ 一致
  ref_motion_phase: 1.0,   # ✅ 一致
  history_actor: 1.0,      # ✅ 一致
}
```

**Sim2Sim配置 (mujoco_config_atom.yaml)**:
```yaml
obs_scale_base_ang_vel: 0.25  # ✅
obs_scale_dof_pos: 1.0        # ✅
obs_scale_dof_vel: 0.05       # ✅
obs_scale_gvec: 1.0           # ✅
obs_scale_refmotion: 1.0      # ✅
obs_scale_hist: 1.0           # ✅
```

**结论**: 观测缩放系数是**一致的** ✅

---

### 3️⃣ **噪声配置差异** ⚠️
**训练配置**:
```yaml
noise_scales: {
  base_ang_vel: 0.3,        # ✅ 一致
  projected_gravity: 0.2,   # ✅ 一致
  dof_pos: 0.01,            # ✅ 一致
  dof_vel: 1.0,             # ✅ 一致
}
```

**Sim2Sim代码 (atom_mujoco.py:88-97)**:
```python
if cfg.use_noise:
    noise_base_ang_vel = (np.random.rand(3) * 2. - 1.) * 0.3      # ✅
    noise_projected_gravity = (np.random.rand(3) * 2. - 1.) * 0.2 # ✅
    noise_dof_pos = (np.random.rand(27) * 2. - 1.) * 0.01         # ✅
    noise_dof_vel = (np.random.rand(27) * 2. - 1.) * 1.0          # ✅
```

**结论**: 噪声配置是**一致的** ✅

---

### 4️⃣ **控制频率不匹配** ⚠️⚠️
**训练配置**: 50 Hz (dt = 0.02s)

**Sim2Sim配置**:
- `simulation_dt: 0.005` (仿真步长)
- `control_decimation: 4` (控制抽取率)
- 实际控制频率: 1 / (0.005 * 4) = **50 Hz** ✅

**结论**: 控制频率是**一致的** ✅

---

### 5️⃣ **动作缩放系数** ✅
**训练**: `action_scale: 0.25`
**Sim2Sim**: `action_scale: 0.25`

**结论**: 动作缩放是**一致的** ✅

---

### 6️⃣ **观测构建顺序问题** ⚠️⚠️⚠️
**代码中的观测顺序 (atom_mujoco.py:123-143)**:
```python
# actions: 0:27
obs_all[0, idx:idx+27] = obs_sigle[0, 0:27].copy()
# base_ang_vel: 27:30
obs_all[0, idx:idx+3] = obs_sigle[0, 27:30].copy()
# dof_pos: 30:57
obs_all[0, idx:idx+27] = obs_sigle[0, 30:57].copy()
# dof_vel: 57:84
obs_all[0, idx:idx+27] = obs_sigle[0, 57:84].copy()
# history_actor: 84:436 (352 维)
obs_all[0, idx:idx+352] = hist_obs_c[0] * cfg.obs_scale_hist
# projected_gravity: 436:439
obs_all[0, idx:idx+3] = obs_sigle[0, 84:87].copy()
# ref_motion_phase: 439:440
obs_all[0, idx] = obs_sigle[0, 87].copy()
```

**训练配置的观测顺序 (deepmimic_a2c_nolinvel_LARGEnoise_history_atom.yaml)**:
```yaml
actor_obs: [
  base_ang_vel,          # 3
  projected_gravity,     # 3
  dof_pos,               # 27
  dof_vel,               # 27
  actions,               # 27
  ref_motion_phase,      # 1
  history_actor          # 352
]
```

**问题**: 观测顺序**完全不对**！❌❌❌

**正确顺序应该是**:
```
base_ang_vel (3) → projected_gravity (3) → dof_pos (27) → dof_vel (27) → actions (27) → ref_motion_phase (1) → history_actor (352)
```

**当前错误顺序**:
```
actions (27) → base_ang_vel (3) → dof_pos (27) → dof_vel (27) → history_actor (352) → projected_gravity (3) → ref_motion_phase (1)
```

---

### 7️⃣ **历史观测更新顺序** ⚠️⚠️
**代码 (atom_mujoco.py:170-173)**:
```python
hist_obs = np.concatenate([
    hist_dict[key].reshape(1, -1)
    for key in hist_dict.keys()  # ❌ dict.keys() 顺序不确定！
], axis=1).astype(np.float32)
```

**问题**: Python字典的`.keys()`顺序在Python 3.7+是插入顺序，但这依赖实现细节，不可靠！

**应该使用**:
```python
history_keys = ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']
hist_obs = np.concatenate([
    hist_dict[key].reshape(1, -1)
    for key in history_keys
], axis=1).astype(np.float32)
```

---

### 8️⃣ **关节角度offset处理** ⚠️
**代码 (atom_mujoco.py:109)**:
```python
dof_pos = mujoco_dof_pos - cfg.default_dof_pos
```

**问题**: 这是对的！但需要确认`default_dof_pos`是否与训练一致。

**训练配置**: 所有默认角度都是0.0
**Sim2Sim配置**: 所有默认角度都是0.0

**结论**: ✅ 一致

---

### 9️⃣ **PKL数据格式问题** ⚠️
**代码 (atom_mujoco.py:439-446)**:
```python
joint_aa = dof[:, None] * dof_axis
num_augment_joint = 5  # left_hand, right_hand, head, left_toe, right_toe
pose_aa = np.concatenate([
    root_rot_vec[None, :],
    joint_aa,
    np.zeros((num_augment_joint, 3), dtype=np.float32)
], axis=0)
```

**问题**: ATOM有33个body，但只有27个DOF，扩展了5个关节。这是正确的！✅

---

### 🔟 **Episode步数与时长配置** ⚠️
**配置**:
- `episode_steps: 140`
- `simulation_dt: 0.005`
- `control_decimation: 4`
- 实际episode时长: 140 * 0.005 * 4 = **2.8秒**
- `cycle_time: 2.8` ✅

**结论**: Episode时长与cycle_time匹配 ✅

---

## 📋 **问题优先级**

### 🚨 **严重问题 (必须修复)**:
1. **观测顺序错误** - 会导致策略完全无法工作
2. **PD gains关节顺序** - 需要仔细核对

### ⚠️ **重要问题**:
3. **历史观测拼接顺序** - 可能导致性能下降

### ✅ **已正确**:
- 观测缩放系数
- 噪声配置
- 控制频率
- 动作缩放
- PKL数据格式
- Episode时长

---

## 🔧 **修复建议**

### 1. 立即修复观测顺序
```python
def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    # ... 前面部分不变 ...
    
    # 完整观测维度：440
    num_obs_full = 3 + 3 + 27 + 27 + 27 + 1 + 352  # 440
    obs_all = np.zeros([1, num_obs_full], dtype=np.float32)
    
    idx = 0
    # base_ang_vel: 0:3
    obs_all[0, idx:idx+3] = obs_sigle[0, 27:30].copy()
    idx += 3
    # projected_gravity: 3:6
    obs_all[0, idx:idx+3] = obs_sigle[0, 84:87].copy()
    idx += 3
    # dof_pos: 6:33
    obs_all[0, idx:idx+27] = obs_sigle[0, 30:57].copy()
    idx += 27
    # dof_vel: 33:60
    obs_all[0, idx:idx+27] = obs_sigle[0, 57:84].copy()
    idx += 27
    # actions: 60:87
    obs_all[0, idx:idx+27] = obs_sigle[0, 0:27].copy()
    idx += 27
    # ref_motion_phase: 87:88
    obs_all[0, idx] = obs_sigle[0, 87].copy()
    idx += 1
    # history_actor: 88:440
    obs_all[0, idx:idx+352] = hist_obs_c[0] * cfg.obs_scale_hist
    
    # ...
```

### 2. 修复历史观测拼接
使用固定的键顺序：
```python
history_keys = ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']
hist_obs = np.concatenate([
    hist_dict[key].reshape(1, -1)
    for key in history_keys
], axis=1).astype(np.float32)
```

### 3. 验证PD gains顺序
确认关节顺序是否为:
```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch, left_elbow_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_elbow_roll, right_wrist_pitch, right_wrist_yaw
```

---

## ✅ **修复后验证清单**

- [ ] 观测顺序与训练配置一致
- [ ] 历史观测拼接顺序固定
- [ ] PD gains关节顺序正确
- [ ] 运行sim2sim测试机器人是否稳定
- [ ] 对比训练时的观测值范围
- [ ] 确认PKL数据格式正确

