# ATOM Sim2Sim 问题总结报告

**检查日期**: 2025-10-15  
**检查文件**: `asap_mujoco_sim/atom_mujoco.py`, `asap_mujoco_sim/atom_config/mujoco_config_atom.yaml`  
**训练配置**: `logs/MotionTracking/gestruepd3+main_rand/config.yaml`

---

## ✅ 已验证正确的配置

### 1. PD Gains
- ✅ 所有27个关节的 Kp 和 Kd 完全匹配
- ✅ 按训练配置的关节顺序正确映射

### 2. 观测缩放
- ✅ `base_ang_vel`: 0.25
- ✅ `dof_pos`: 1.0  
- ✅ `dof_vel`: 0.05
- ✅ `projected_gravity`: 1.0

### 3. 噪声参数
- ✅ `base_ang_vel`: 0.3
- ✅ `projected_gravity`: 0.2
- ✅ `dof_pos`: 0.01
- ✅ `dof_vel`: 1.0

### 4. 控制频率
- ✅ 训练: 200Hz 仿真 / 4 抽取 = 50Hz 控制
- ✅ Sim2Sim: 200Hz 仿真 / 4 抽取 = 50Hz 控制

### 5. 动作缩放
- ✅ `action_scale`: 0.25

### 6. 观测顺序
- ✅ 完整观测按字母序排列匹配训练配置:
  - [0:27] actions
  - [27:30] base_ang_vel
  - [30:57] dof_pos
  - [57:84] dof_vel
  - [84:436] history_actor
  - [436:439] projected_gravity
  - [439:440] ref_motion_phase

---

## ❌ 发现的严重问题

### 🚨 问题 1: 历史观测中包含噪声 (严重BUG)

#### 问题描述
当前代码在构建单帧观测时添加了噪声，然后将**带噪声的观测**加入历史。但训练时历史观测是**不加噪声**的！

#### 影响分析
- ⚠️ **策略接收到的历史观测分布与训练时完全不一致**
- ⚠️ **导致策略产生错误的动作**
- ⚠️ **可能导致机器人不稳定、更容易摔倒/早停**

#### 训练配置证据
```yaml
# logs/MotionTracking/gestruepd3+main_rand/config.yaml
obs:
  noise_scales:
    base_ang_vel: 0.3      # 当前帧加噪声
    projected_gravity: 0.2  # 当前帧加噪声
    dof_pos: 0.01          # 当前帧加噪声
    dof_vel: 1.0           # 当前帧加噪声
    actions: 0.0           # 不加噪声
    history_actor: 0.0     # ❗历史不加噪声
    history_critic: 0.0    # ❗历史不加噪声
    ref_motion_phase: 0.0  # 不加噪声
```

#### 当前错误实现

```python
# asap_mujoco_sim/atom_mujoco.py (第76-148行)
def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    # ...
    
    # 步骤1: 构建单帧观测，添加噪声
    if cfg.use_noise:
        noise_base_ang_vel = (np.random.rand(3) * 2. - 1.) * 0.3
        noise_dof_pos = (np.random.rand(27) * 2. - 1.) * 0.01
        noise_dof_vel = (np.random.rand(27) * 2. - 1.) * 1.0
        noise_projected_gravity = (np.random.rand(3) * 2. - 1.) * 0.2
    
    obs_sigle = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
    obs_sigle[0, 0:27] = action  # ✅ action不加噪声
    obs_sigle[0, 27:30] = (mujoco_base_angvel + noise_base_ang_vel) * cfg.obs_scale_base_ang_vel  # ❌ 带噪声
    obs_sigle[0, 30:57] = (dof_pos + noise_dof_pos) * cfg.obs_scale_dof_pos  # ❌ 带噪声
    obs_sigle[0, 57:84] = (mujoco_dof_vel + noise_dof_vel) * cfg.obs_scale_dof_vel  # ❌ 带噪声
    obs_sigle[0, 84:87] = (mujoco_gvec + noise_projected_gravity) * cfg.obs_scale_gvec  # ❌ 带噪声
    obs_sigle[0, 87] = ref_motion_phase * cfg.obs_scale_refmotion  # ✅ 不加噪声
    
    # ...
    
    # 步骤2: 使用带噪声的 obs_sigle 更新历史 ❌❌❌
    hist_obs_cat = update_hist_obs(hist_dict, obs_sigle)  # BUG: 历史中包含噪声！
    
    return obs_all, hist_obs_cat
```

#### 修复方案 (推荐)

```python
def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    """
    构建 ATOM 机器人观测（27 DOF）
    修复: 历史观测不加噪声，当前帧观测加噪声
    """
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    
    ref_motion_phase = (counter + 1) * cfg.simulation_dt / cfg.cycle_time
    ref_motion_phase = np.clip(ref_motion_phase, 0, 1)
    
    # 1️⃣ 构建无噪声的单帧观测（用于更新历史）
    obs_clean = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
    obs_clean[0, 0:27] = action
    obs_clean[0, 27:30] = mujoco_base_angvel * cfg.obs_scale_base_ang_vel
    dof_pos = mujoco_dof_pos - cfg.default_dof_pos
    obs_clean[0, 30:57] = dof_pos * cfg.obs_scale_dof_pos
    obs_clean[0, 57:84] = mujoco_dof_vel * cfg.obs_scale_dof_vel
    obs_clean[0, 84:87] = mujoco_gvec * cfg.obs_scale_gvec
    obs_clean[0, 87] = ref_motion_phase * cfg.obs_scale_refmotion
    
    # 2️⃣ 使用无噪声观测更新历史
    hist_obs_cat = update_hist_obs(hist_dict, obs_clean)
    
    # 3️⃣ 生成噪声（如果启用）
    if cfg.use_noise:
        noise_base_ang_vel = (np.random.rand(3) * 2. - 1.) * 0.3
        noise_projected_gravity = (np.random.rand(3) * 2. - 1.) * 0.2
        noise_dof_pos = (np.random.rand(27) * 2. - 1.) * 0.01
        noise_dof_vel = (np.random.rand(27) * 2. - 1.) * 1.0
    else:
        noise_base_ang_vel = np.zeros(3)
        noise_projected_gravity = np.zeros(3)
        noise_dof_pos = np.zeros(27)
        noise_dof_vel = np.zeros(27)
    
    # 4️⃣ 组装完整观测（当前帧用有噪声版本，历史无噪声）
    obs_all = np.zeros([1, 440], dtype=np.float32)
    
    # actions: 0:27 (无噪声)
    obs_all[0, 0:27] = action
    
    # base_ang_vel: 27:30 (有噪声)
    obs_all[0, 27:30] = (mujoco_base_angvel + noise_base_ang_vel) * cfg.obs_scale_base_ang_vel
    
    # dof_pos: 30:57 (有噪声)
    obs_all[0, 30:57] = (dof_pos + noise_dof_pos) * cfg.obs_scale_dof_pos
    
    # dof_vel: 57:84 (有噪声)
    obs_all[0, 57:84] = (mujoco_dof_vel + noise_dof_vel) * cfg.obs_scale_dof_vel
    
    # history_actor: 84:436 (无噪声！✅)
    obs_all[0, 84:436] = hist_obs_cat[0] * cfg.obs_scale_hist
    
    # projected_gravity: 436:439 (有噪声)
    obs_all[0, 436:439] = (mujoco_gvec + noise_projected_gravity) * cfg.obs_scale_gvec
    
    # ref_motion_phase: 439:440 (无噪声)
    obs_all[0, 439] = ref_motion_phase * cfg.obs_scale_refmotion
    
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)
    
    return obs_all, hist_obs_cat
```

---

## ⚠️ 次要问题

### 问题 2: 字典键顺序依赖

#### 问题描述
`update_hist_obs` 函数中使用 `hist_dict.keys()` 来拼接历史观测，依赖字典的插入顺序。

#### 当前实现 (第170-174行)
```python
def update_hist_obs(hist_dict, obs_sigle):
    # ...
    hist_obs = np.concatenate([
        hist_dict[key].reshape(1, -1)
        for key in hist_dict.keys()  # ⚠️ 依赖字典顺序
    ], axis=1).astype(np.float32)
    return hist_obs
```

#### 影响
- Python 3.7+ 字典保证插入顺序，所以实际上是安全的
- 但为了代码可读性和健壮性，建议显式指定键顺序

#### 建议修复
```python
def update_hist_obs(hist_dict, obs_sigle):
    slices = {
        'actions': slice(0, 27),
        'base_ang_vel': slice(27, 30),
        'dof_pos': slice(30, 57),
        'dof_vel': slice(57, 84),
        'projected_gravity': slice(84, 87),
        'ref_motion_phase': slice(87, 88)
    }

    for key, slc in slices.items():
        arr = np.delete(hist_dict[key], -1, axis=0)
        arr = np.vstack((obs_sigle[0, slc], arr))
        hist_dict[key] = arr

    # 显式指定键顺序（按字母序，与训练一致）
    history_keys = ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']
    hist_obs = np.concatenate([
        hist_dict[key].reshape(1, -1)
        for key in history_keys  # ✅ 显式顺序
    ], axis=1).astype(np.float32)
    return hist_obs
```

---

## 📊 早停行为分析

### 问题: 为什么 ATOM 可能比 G1 更容易早停？

检查发现早停逻辑和阈值都相同:
- ✅ 早停函数逻辑相同
- ✅ 重力投影计算相同  
- ✅ 早停阈值相同 (0.85, 对应 ~58°倾斜)

**可能的原因**:

1. **历史观测噪声问题** (上述BUG)
   - 训练时历史无噪声，sim2sim有噪声
   - 导致策略输出不稳定动作
   - 机器人更容易失去平衡

2. **机器人动力学差异**
   - ATOM 和 G1 质量分布不同
   - 惯性特性不同
   - 对相同控制输入的响应不同

3. **初始姿态不同**
   - 可能导致不同的稳定性

4. **策略训练质量**
   - 需要检查 ATOM 策略的训练时长、收敛情况

### 调试建议

在 `check_termination` 中添加详细日志:

```python
def check_termination(mujoco_data, cfg, counter):
    if not cfg.use_termination:
        return False

    mujoco_gvec = mujoco_data["mujoco_gvec"]
    gravity_x_violation = abs(mujoco_gvec[0]) > cfg.termination_gravity_x
    gravity_y_violation = abs(mujoco_gvec[1]) > cfg.termination_gravity_y
    gravity_termination = gravity_x_violation or gravity_y_violation

    # 每10步打印一次状态
    if counter % 10 == 0:
        print(f"[Step {counter:4d}] gvec=[{mujoco_gvec[0]:+.3f}, {mujoco_gvec[1]:+.3f}, {mujoco_gvec[2]:+.3f}] "
              f"|gx|={abs(mujoco_gvec[0]):.3f} (thresh={cfg.termination_gravity_x}) "
              f"|gy|={abs(mujoco_gvec[1]):.3f} (thresh={cfg.termination_gravity_y})")

    should_terminate = gravity_termination

    if should_terminate:
        print(f"[Termination] Step {counter}: "
              f"gravity_x={mujoco_gvec[0]:.3f}(>{cfg.termination_gravity_x}), "
              f"gravity_y={mujoco_gvec[1]:.3f}(>{cfg.termination_gravity_y})")

    return should_terminate
```

---

## 🔧 修复优先级

1. **🚨 高优先级**: 修复历史观测噪声问题 (问题1)
   - 这是导致策略失效的最可能原因
   - 必须立即修复

2. **⚠️ 中优先级**: 显式指定历史观测键顺序 (问题2)
   - 提高代码健壮性和可读性
   - 防止未来的潜在问题

3. **💡 建议**: 添加调试日志
   - 帮助理解早停行为
   - 分析策略表现

---

## 📝 总结

经过详细检查，发现了一个**严重的BUG**: 历史观测中包含噪声，而训练时历史是不加噪声的。这会导致策略接收到错误的观测分布，可能是导致早停和不稳定的主要原因。

**修复后预期效果**:
- ✅ 策略接收到与训练时一致的观测分布
- ✅ 动作输出更加稳定和准确
- ✅ 减少不必要的早停
- ✅ 提高轨迹采集质量

**下一步**:
1. 应用推荐的修复代码
2. 运行 sim2sim 测试
3. 对比修复前后的早停率和轨迹质量

