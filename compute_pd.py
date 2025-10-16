import math


def calculate_parameters():
    # 给定的参数
    ARMATURE_HIP_PITCH = 0.10897986
    ARMATURE_HIP_ROLL = 0.070622376
    ARMATURE_HIP_YAW = 0.036700268
    ARMATURE_KNEE = 0.0812586105
    ARMATURE_ANKLE = 0.01536
    ARMATURE_WAIST = 0.1123945708

    # joint_armature = {
    #     "left_hip_pitch_joint": 0.10897986,
    #     "left_hip_roll_joint": 0.070622376,
    #     "left_hip_yaw_joint": 0.036700268,
    #     "left_knee_joint": 0.0812586105,
    #     "left_ankle_pitch_joint": 0.01536,
    #     "left_ankle_roll_joint": 0.01536,

    #     "right_hip_pitch_joint": 0.10897986,
    #     "right_hip_roll_joint": 0.070622376,
    #     "right_hip_yaw_joint": 0.036700268,
    #     "right_knee_joint": 0.0812586105,
    #     "right_ankle_pitch_joint": 0.01536,
    #     "right_ankle_roll_joint": 0.01536,

    #     "waist_yaw_joint": 0.1123945708,

    #     "left_shoulder_pitch_joint": 24.001,
    #     "left_shoulder_roll_joint": 24.001,
    #     "left_shoulder_yaw_joint": 24.001,
    #     "left_elbow_pitch_joint": 24.001,
    #     "left_elbow_roll_joint": 7.302,
    #     "left_wrist_pitch_joint": 7.302,
    #     "left_wrist_yaw_joint": 7.302,

    #     "right_shoulder_pitch_joint": 24.001,
    #     "right_shoulder_roll_joint": 24.001,
    #     "right_shoulder_yaw_joint": 24.001,
    #     "right_elbow_pitch_joint": 24.001,
    #     "right_elbow_roll_joint": 7.302,
    #     "right_wrist_pitch_joint": 7.302,
    #     "right_wrist_yaw_joint": 7.302,

    # }

    NATURAL_FREQ_HZ = 10
    NATURAL_FREQ = NATURAL_FREQ_HZ * 2.0 * math.pi  # 转换为角频率 (rad/s)
    DAMPING_RATIO = 2

    # 计算刚度 (Stiffness = Armature * ω²)
    STIFFNESS_HIP_PITCH = ARMATURE_HIP_PITCH * NATURAL_FREQ ** 2
    STIFFNESS_HIP_ROLL = ARMATURE_HIP_ROLL * NATURAL_FREQ ** 2
    STIFFNESS_HIP_YAW = ARMATURE_HIP_YAW * NATURAL_FREQ ** 2
    STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ ** 2
    STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ ** 2
    STIFFNESS_WAIST = ARMATURE_WAIST * NATURAL_FREQ ** 2

    # 计算阻尼 (Damping = 2 * ζ * Armature * ω)
    DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_HIP_PITCH * NATURAL_FREQ
    DAMPING_HIP_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_HIP_ROLL * NATURAL_FREQ
    DAMPING_HIP_YAW = 2.0 * DAMPING_RATIO * ARMATURE_HIP_YAW * NATURAL_FREQ
    DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ
    DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
    DAMPING_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_WAIST * NATURAL_FREQ

    ACTION_SCALE_HIP_PITCH = 0.25 * 280.0 / STIFFNESS_HIP_PITCH
    ACTION_SCALE_HIP_ROLL = 0.25 * 308.0 / STIFFNESS_HIP_ROLL
    ACTION_SCALE_HIP_YAW = 0.25 * 140.0 / STIFFNESS_HIP_YAW
    ACTION_SCALE_KNEE = 0.25 * 360.0 / STIFFNESS_KNEE
    ACTION_SCALE_ANKLE = 0.25 * 130.0 / STIFFNESS_ANKLE

    # 打印基础参数
    print("=" * 60)
    print("基础参数:")
    print("=" * 60)
    print(f"自然频率 (ω): {NATURAL_FREQ_HZ} HZ")
    print(f"自然频率 (ω): {NATURAL_FREQ:.6f} rad/s")
    print(f"阻尼比 (ζ): {DAMPING_RATIO}")
    print()

    # 打印电机参数
    print("=" * 60)
    print("电机参数:")
    print("=" * 60)
    print(f"ARMATURE_HIP_PITCH: {ARMATURE_HIP_PITCH:.8f} kg·m²")
    print(f"ARMATURE_HIP_ROLL: {ARMATURE_HIP_ROLL:.8f} kg·m²")
    print(f"ARMATURE_HIP_YAW: {ARMATURE_HIP_YAW:.8f} kg·m²")
    print(f"ARMATURE_KNEE: {ARMATURE_KNEE:.8f} kg·m²")
    print(f"ARMATURE_ANKLE: {ARMATURE_ANKLE:.8f} kg·m²")
    print(f"ARMATURE_WAIST: {ARMATURE_WAIST:.8f} kg·m²")
    print()

    # 打印刚度计算结果
    print("=" * 60)
    print("刚度参数 (Stiffness):")
    print("=" * 60)
    print(f"STIFFNESS_HIP_PITCH: {STIFFNESS_HIP_PITCH:.2f} N·m/rad")
    print(f"STIFFNESS_HIP_ROLL: {STIFFNESS_HIP_ROLL:.2f} N·m/rad")
    print(f"STIFFNESS_HIP_YAW: {STIFFNESS_HIP_YAW:.2f} N·m/rad")
    print(f"STIFFNESS_KNEE: {STIFFNESS_KNEE:.2f} N·m/rad")
    print(f"STIFFNESS_ANKLE: {STIFFNESS_ANKLE:.2f} N·m/rad")
    print(f"STIFFNESS_WAIST: {STIFFNESS_WAIST:.8f} kg·m²")
    print()

    # 打印阻尼计算结果
    print("=" * 60)
    print("阻尼参数 (Damping):")
    print("=" * 60)
    print(f"DAMPING_HIP_PITCH: {DAMPING_HIP_PITCH:.2f} N·m·s/rad")
    print(f"DAMPING_HIP_ROLL: {DAMPING_HIP_ROLL:.2f} N·m·s/rad")
    print(f"DAMPING_HIP_YAW: {DAMPING_HIP_YAW:.2f} N·m·s/rad")
    print(f"DAMPING_KNEE: {DAMPING_KNEE:.2f} N·m·s/rad")
    print(f"DAMPING_ANKLE: {DAMPING_ANKLE:.2f} N·m·s/rad")
    print(f"STIFFNESS_WAIST: {DAMPING_WAIST:.8f} kg·m²")
    print()

    # 打印 action scale 计算结果
    print("=" * 60)
    print("action scale:")
    print("=" * 60)
    print(f"ACTION_SCALE_HIP_PITCH: {ACTION_SCALE_HIP_PITCH:.2f} ")
    print(f"ACTION_SCALE_HIP_ROLL: {ACTION_SCALE_HIP_ROLL:.2f} ")
    print(f"ACTION_SCALE_HIP_YAW: {ACTION_SCALE_HIP_YAW:.2f} ")
    print(f"ACTION_SCALE_KNEE: {ACTION_SCALE_KNEE:.2f} ")
    print(f"ACTION_SCALE_ANKLE: {ACTION_SCALE_ANKLE:.2f} ")
    print()


if __name__ == "__main__":
    calculate_parameters()