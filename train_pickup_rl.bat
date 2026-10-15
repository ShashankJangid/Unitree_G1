@echo off
title Unitree G1 Whole-Body Bending Down and Pickup RL Training
cd /d "%~dp0g1_physics_sim"
"..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe" train_g1_pickup_rl.py %*
pause
