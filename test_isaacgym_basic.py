#!/usr/bin/env python
"""
测试 IsaacGym 基础功能（不创建 viewer）
"""
import os
os.environ['LD_LIBRARY_PATH'] = '/home/dobot/miniforge3/envs/hvgym/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

print("测试 IsaacGym 基础功能（headless）...")

from isaacgym import gymapi

gym = gymapi.acquire_gym()
print("✓ gym 实例创建成功")

sim_params = gymapi.SimParams()
sim_params.dt = 1.0/60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

# headless: graphics_device_id = -1
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
print("✓ 模拟环境创建成功（headless）")

# 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)
print("✓ 地面添加成功")

# 运行几步模拟
for i in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

print("✓ 模拟运行成功")
gym.destroy_sim(sim)
print("\n✓✓✓ 所有基础测试通过！IsaacGym 本身工作正常。")
print("问题确定在图形渲染部分。")

