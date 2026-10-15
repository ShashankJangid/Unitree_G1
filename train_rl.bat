@echo off
title Unitree G1 29-DOF RL Locomotion Trainer
cd /d "%~dp0g1_physics_sim"
"..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe" train_g1_loco_rl.py %*
pause
