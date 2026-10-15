@echo off
title Unitree G1 Official Pretrained RL Policy
cd /d "%~dp0g1_physics_sim"
"..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe" play_g1_official_rl.py %*
pause
