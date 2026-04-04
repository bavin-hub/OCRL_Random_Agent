we need 23 or 29 dof no less than that

# supports only legged motion (damn mf...should have noticed this)
1.https://github.com/julienokumu/unitree_rl_mugym (not using this)


# supports 29 dof motion
2.https://github.com/unitreerobotics/unitree_rl_mjlab

    -> this one works
    -> has saved policies over 10000 iterations (should help in finding both good and bad)
    -> after cloing our RandomAgent repo follow the above link to setup the unitree_rl_mjlab package
    -> trace the script base.py : scripts/play.py (click on NativeMujocoViewer) ->  viewer.py (click on base viewer) -> base.py
    -> copy the content of base_mod.py file (in the data dir of RandomAgent) and paste in the traced base.py file
    -> a function has be written in base_mod.py that samples traj. (for both 23 and 29 dof) and saves them into a db
    -> pass the db dir in
    -> command to run the baseline policy : python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-03-16_22-57-16/model_10000.pt
    -> the db should be saved in the dir. you passed

miniconda3/envs/unitree_rl_mjlab/lib/python3.11/site-packages/mjlab/viewer/base.py

# for motion retargetting (GVHMR + ASAP)...need to update the command flow 
3. https://github.com/LeCAR-Lab/ASAP

