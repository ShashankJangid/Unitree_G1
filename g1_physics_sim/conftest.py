"""Make g1_physics_sim modules importable regardless of the pytest invocation
directory, so `pytest test_sdk_contract.py` and `pytest g1_physics_sim/...`
behave identically."""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))