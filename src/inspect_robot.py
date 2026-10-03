from pathlib import Path

import mujoco

MODEL_PATH = Path.home() / "Documents/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

print("\n=== BODIES ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )
    print(i, name)

print("\n=== SITES ===")
for i in range(model.nsite):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        i
    )
    print(i, name)

print("\n=== JOINTS ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )
    print(i, name)
