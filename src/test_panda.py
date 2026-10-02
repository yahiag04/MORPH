import time
import numpy as np
import mujoco
import mujoco.viewer

from simulation.panda_env import PandaEnv

env = PandaEnv()

print("Number of joints:", env.model.njnt)
print("Number of actuators:", env.model.nu)
print("Actuators:")

for i in range(env.model.nu):
    print(i, mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    start = time.time()
    last_position_print = -0.5

    while viewer.is_running():

        t = time.time() - start

        # Move joint 1 sinusoidally
        env.data.ctrl[0] = 0.4 * np.sin(t)

        env.step()

        if t - last_position_print >= 0.5:
            position = env.get_end_effector_position()
            print(f"Hand XYZ: {position.round(4).tolist()}", flush=True)
            last_position_print = t

        viewer.sync()

        time.sleep(env.model.opt.timestep)
