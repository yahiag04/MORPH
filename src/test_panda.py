import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "/Users/yahiaghallale/Documents/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Actuators:")

for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running():

        t = time.time() - start

        # Move joint 1 sinusoidally
        data.ctrl[0] = 0.4 * np.sin(t)

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)