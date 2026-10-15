import sys
import pathlib

root_dir = pathlib.Path(__file__).resolve().parent
sim_dir = root_dir / "g1_physics_sim"
sys.path.insert(0, str(sim_dir))

from play_g1_official_rl import main

if __name__ == "__main__":
    main()
