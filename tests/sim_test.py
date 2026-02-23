import mujoco
from robot_descriptions import g1_description
import mujoco.viewer
import numpy as np 
import time


model_path = '/home/bavin/cmu/sem2/ocrl/OCRL_Random_Agent/tests/mujoco_menagerie/unitree_go1/scene.xml'
# model_path = '/home/bavin/cmu/sem2/ocrl/OCRL_Random_Agent/tests/mujoco_menagerie/unitree_g1/scene.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        if model.nu > 0:
            # data.ctrl[:] = 0.2 * np.sin(time.time())
            ctrl = np.random.uniform(-1, 1, size=12)
            data.ctrl = ctrl

        # print(data)
        # print(model.nu)
        print('\n\n')
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)



'''
Get robot state

data.qpos
data.qvel
data.ctrl (output command)
data.contact
body_id = model.body("base").id
pos = data.xpos[body_id].copy()     
quat = data.xquat[body_id].copy() 


To-Do

1. figure out the joint range (rads)....basically prune the robot and understand it
2. write a control loop gym kinda env (any baseline controller)
3. add assets to the env (diverse env)

'''


