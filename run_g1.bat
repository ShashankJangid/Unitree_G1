@echo off
title Unitree G1 Simulation
cd /d "%~dp0g1_physics_sim"
"..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe" g1_sandbox.py
pause
