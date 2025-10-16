import math
from typing import Dict, Tuple

# =========================
# 配置区
# =========================
NATURAL_FREQ_HZ = 10.0        # 自然频率（Hz）
DAMPING_RATIO   = 2.0         # 阻尼比 ζ
OMEGA           = 2.0 * math.pi * NATURAL_FREQ_HZ  # rad/s

# 关节端“等效转动惯量”（kg·m^2）
# 这些数值应是 motor inertia 通过减速比反射后的等效惯量（I_joint = I_motor * gear^2）
# 下面这份就是你注释里那张字典（已解注释）。
joint_armature: Dict[str, float] = {
    # 左腿
    "left_hip_pitch_joint":   0.10897986,
    "left_hip_roll_joint":    0.070622376,
    "left_hip_yaw_joint":     0.036700268,
    "left_knee_joint":        0.0812586105,
    "left_ankle_pitch_joint": 0.01536,
    "left_ankle_roll_joint":  0.01536,

    # 右腿
    "right_hip_pitch_joint":   0.10897986,
    "right_hip_roll_joint":    0.070622376,
    "right_hip_yaw_joint":     0.036700268,
    "right_knee_joint":        0.0812586105,
    "right_ankle_pitch_joint": 0.01536,
    "right_ankle_roll_joint":  0.01536,

    # 腰
    "waist_yaw_joint": 0.1123945708,

    # 左臂
    "left_shoulder_pitch_joint": 24.001,
    "left_shoulder_roll_joint":  24.001,
    "left_shoulder_yaw_joint":   24.001,
    "left_elbow_pitch_joint":    24.001,
    "left_elbow_roll_joint":     7.302,
    "left_wrist_pitch_joint":    7.302,
    "left_wrist_yaw_joint":      7.302,

    # 右臂
    "right_shoulder_pitch_joint": 24.001,
    "right_shoulder_roll_joint":  24.001,
    "right_shoulder_yaw_joint":   24.001,
    "right_elbow_pitch_joint":    24.001,
    "right_elbow_roll_joint":     7.302,
    "right_wrist_pitch_joint":    7.302,
    "right_wrist_yaw_joint":      7.302,
}

# =========================
# 可选：如果只有“电机转子惯量 + 减速比”，可用此函数先算等效惯量
# =========================
def reflect_inertia(motor_inertia: float, gear_ratio: float) -> float:
    """I_joint = I_motor * gear^2"""
    return motor_inertia * (gear_ratio ** 2)

# =========================
# 计算函数
# =========================
def kp_kd_from_armature(I_joint: float, omega: float = OMEGA, zeta: float = DAMPING_RATIO) -> Tuple[float, float]:
    """
    给定关节端等效惯量 I_joint，计算 kp/kd
    kp = I * ω^2
    kd = 2 * I * ζ * ω
    """
    kp = I_joint * (omega ** 2)
    kd = 2.0 * I_joint * zeta * omega
    return kp, kd

def main():
    print("=" * 72)
    print(f"自然频率: {NATURAL_FREQ_HZ:.3f} Hz  |  ω = {OMEGA:.6f} rad/s  |  阻尼比 ζ = {DAMPING_RATIO:.3f}")
    print("=" * 72)
    header = f"{'joint':<30} {'I_joint(kg·m^2)':>16} {'kp(N·m/rad)':>16} {'kd(N·m·s/rad)':>18}"
    print(header)
    print("-" * len(header))

    results: Dict[str, Dict[str, float]] = {}

    for name in sorted(joint_armature.keys()):
        I = float(joint_armature[name])
        kp, kd = kp_kd_from_armature(I)
        results[name] = {"I": I, "kp": kp, "kd": kd}
        print(f"{name:<30} {I:>16.6f} {kp:>16.2f} {kd:>18.2f}")

    # 便于直接拷贝到配置里的 YAML 片段
    print("\n# ================= YAML 片段（kps/kds，以关节名映射）=================")
    print("stiffness:  # N·m/rad")
    for name in sorted(results.keys()):
        print(f"  {name}: {results[name]['kp']:.6f}")
    print("\ndamping:    # N·m·s/rad")
    for name in sorted(results.keys()):
        print(f"  {name}: {results[name]['kd']:.6f}")

if __name__ == '__main__':
    main()
