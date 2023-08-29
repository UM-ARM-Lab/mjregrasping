import json
import pickle
import time
from pathlib import Path

import mujoco
import pysdf_tools

from mjregrasping.movie import MjMovieMaker
from mjregrasping.scenarios import Scenario


def save_trial(i, phy, scenario, sdf_path, skeletons):
    trials_root = Path("trial_data") / scenario.name
    phy_path = trials_root / f"{scenario.name}_{i}_phy.pkl"
    with phy_path.open('wb') as f:
        pickle.dump(phy, f)

    trial_info = {
        'phy_path':  phy_path,
        'sdf_path':  sdf_path,
        'skeletons': skeletons,
    }
    trial_path = trials_root / f"{scenario.name}_{i}.pkl"
    with trial_path.open("wb") as f:
        pickle.dump(trial_info, f)


def load_trial(i: int, scenario: Scenario, viz):
    trials_root = Path("trial_data") / scenario.name
    trial_path = trials_root / f"{scenario.name}_{i}.pkl"
    with trial_path.open("rb") as f:
        trial_info = pickle.load(f)
    phy_path = trial_info['phy_path']
    sdf_path = trial_info['sdf_path']
    skeletons = trial_info['skeletons']
    with phy_path.open("rb") as f:
        phy = pickle.load(f)
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)
    if sdf_path:
        sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(sdf_path))
        # viz_slices(sdf)
    else:
        sdf = None
    viz.skeletons(skeletons)

    mov = MjMovieMaker(phy.m)

    now = int(time.time())

    results_root = Path("results") / scenario.name / f"{now}_{i}"
    results_root.mkdir(exist_ok=True, parents=True)

    mov_path = results_root / f'{scenario.name}_{now}_{i}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path)

    metrics_path = results_root / f'{scenario.name}_{now}_{i}.json'

    return phy, sdf, skeletons, mov, metrics_path


def save_metrics(metrics_path: Path, mov: MjMovieMaker, **metrics):
    mov.close()
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2)
