@echo off
title Unitree G1 Simulation
cd /d "%~dp0"
"..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe" g1_sandbox.py
pause