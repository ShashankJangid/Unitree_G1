import sys
import pathlib

sim_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(sim_dir))

from play_g1_official_rl import main

if __name__ == "__main__":
    main()
