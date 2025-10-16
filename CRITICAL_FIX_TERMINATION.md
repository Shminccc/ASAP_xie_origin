# 🚨 ATOM 早停失效问题修复

## 问题描述

**症状**: ATOM 机器人快要摔倒了但不触发早停，导致机器人完全倒地才结束 episode。

## 根本原因

代码在检查早停条件时使用的是**过时的机器人状态**！

### 错误的执行流程

```python
for step in range(cfg.episode_steps * cfg.control_decimation):
    mujoco_data = get_mujoco_data(data)  # ① 获取状态 (t时刻)
    
    tau = pd_control(...)                 # ② 计算力矩
    data.ctrl[:] = tau                    # ③ 设置控制
    mujoco.mj_step(model, data)          # ④ 仿真一步 → 状态更新到 (t+dt)
    
    if counter % cfg.control_decimation == 0:
        # ❌ 使用的是 t 时刻的旧状态!
        obs_buff, hist_obs_c = get_obs(..., mujoco_data, ...)
        should_terminate = check_termination(mujoco_data, ...)  # ❌ 检查旧状态!
        
        # 但保存数据时用的是新状态
        q = data.qpos  # ✅ 这是 t+dt 时刻的状态
```

### 问题分析

1. **`mujoco_data`** 在 `mujoco.mj_step()` **之前** 获取 (t 时刻)
2. **`mujoco.mj_step()`** 执行后，机器人状态已经更新到 (t+dt 时刻)
3. **`check_termination()`** 检查的还是 t 时刻的 `mujoco_data`
4. 结果：**早停检查延迟一个仿真步！**

### 影响

- 机器人倾斜角度已经超过阈值，但系统看到的是上一步的角度
- 当下一步检查时，机器人可能已经完全倒地
- 导致收集到大量无用的"摔倒"轨迹

## 修复方案

在 `mujoco.mj_step()` 之后**重新获取最新状态**用于观测和早停检查。

### 修复后的正确流程

```python
for step in range(cfg.episode_steps * cfg.control_decimation):
    mujoco_data = get_mujoco_data(data)  # ① 获取状态 (t时刻) 用于PD控制
    
    tau = pd_control(...)                 # ② 计算力矩
    data.ctrl[:] = tau                    # ③ 设置控制
    mujoco.mj_step(model, data)          # ④ 仿真一步 → 状态更新到 (t+dt)
    
    if counter % cfg.control_decimation == 0:
        # ✅ 重新获取最新状态!
        mujoco_data = get_mujoco_data(data)  # ⑤ 获取 t+dt 时刻的最新状态
        
        obs_buff, hist_obs_c = get_obs(..., mujoco_data, ...)  # ✅ 使用最新状态
        should_terminate = check_termination(mujoco_data, ...)  # ✅ 检查最新状态
```

### 代码修改

**文件**: `asap_mujoco_sim/atom_mujoco.py`  
**位置**: 第 408-431 行

```diff
         for step in range(cfg.episode_steps * cfg.control_decimation):
             mujoco_data = get_mujoco_data(data)
 
             tau = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"],
                              np.zeros_like(cfg.kds), mujoco_data["mujoco_dof_vel"], cfg)
             tau_limit = np.array(cfg.tau_limit)
             tau = np.clip(tau, -tau_limit, tau_limit)
             mujoco_data['torques'] = tau
 
             data.ctrl[:] = tau
             mujoco.mj_step(model, data)
 
             if counter % cfg.control_decimation == 0:
                 current_step += 1
+                # ✅ 重新获取最新状态用于观测和早停检查
+                mujoco_data = get_mujoco_data(data)
+                
                 obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg)
                 policy_input = {policy.get_inputs()[0].name: obs_buff}
                 action = policy.run(["action"], policy_input)[0]
                 action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)
                 target_dof_pos = action * cfg.action_scale + cfg.default_dof_pos
 
                 should_terminate = check_termination(mujoco_data, cfg, counter)
```

## 为什么 G1 也有这个问题？

检查 `mujoco_track_with_processing.py` 发现 **G1 版本也有同样的问题**！

但为什么 G1 能工作而 ATOM 不行？可能的原因：

1. **G1 策略更鲁棒**: G1 训练得更好，不容易接近早停阈值
2. **G1 动力学更稳定**: G1 机器人本身更难失去平衡
3. **G1 的早停阈值设置更宽松**: 允许更大的倾斜角度

### 建议

**同时修复 G1 版本的代码**，确保早停检查使用最新状态：

```python
# asap_mujoco_sim/mujoco_track_with_processing.py
# 在 mujoco.mj_step(model, data) 之后添加同样的修复
if counter % cfg.control_decimation == 0:
    current_step += 1
    # ✅ 重新获取最新状态
    mujoco_data = get_mujoco_data(data)
    
    obs_buff, hist_obs_c = get_obs(...)
    # ...
```

## 验证方法

修复后，可以通过以下方式验证：

1. **添加调试输出** 查看重力投影值:
   ```python
   def check_termination(mujoco_data, cfg, counter):
       if not cfg.use_termination:
           return False
       
       mujoco_gvec = mujoco_data["mujoco_gvec"]
       
       # 每10步打印一次
       if counter % 10 == 0:
           print(f"[Step {counter:4d}] gvec=[{mujoco_gvec[0]:+.3f}, {mujoco_gvec[1]:+.3f}, {mujoco_gvec[2]:+.3f}] "
                 f"|gx|={abs(mujoco_gvec[0]):.3f} (thresh={cfg.termination_gravity_x}), "
                 f"|gy|={abs(mujoco_gvec[1]):.3f} (thresh={cfg.termination_gravity_y})")
       
       gravity_x_violation = abs(mujoco_gvec[0]) > cfg.termination_gravity_x
       gravity_y_violation = abs(mujoco_gvec[1]) > cfg.termination_gravity_y
       gravity_termination = gravity_x_violation or gravity_y_violation
       
       return gravity_termination
   ```

2. **观察早停率**: 修复后应该能看到更及时的早停，减少完全倒地的情况

3. **检查轨迹质量**: 处理后的轨迹应该有更高的保留率（因为早停及时，不会完全失败）

## 预期效果

修复后：
- ✅ 早停能及时触发，避免机器人完全倒地
- ✅ 收集到的轨迹质量更高
- ✅ 减少无效数据（摔倒后的挣扎）
- ✅ 提高数据处理的效率（更多 episode 达到最小长度要求）

## 总结

这是一个**严重的BUG**，导致早停机制完全失效。修复方法很简单：在每个控制步开始时重新获取机器人的最新状态，用于构建观测和检查早停条件。

**修复优先级**: 🚨 **最高** - 必须立即修复，否则无法有效采集轨迹。
