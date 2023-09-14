import json
from pathlib import Path

import numpy as np
import pandas as pd


def list_mean(x):
    return np.mean([np.mean(l) for l in x])


def avg_grasps(x):
    return np.mean([len(x_i) for x_i in x])

def std_grasps(x):
    return np.std([len(x_i) for x_i in x])


def main():
    print_results_table(load_data([
        Path("results/Untangle/untangle_ours_v3"),
        Path("results/Untangle/untangle_no_signature_v1"),
        Path("results/Untangle/untangle_tamp5_v2"),
        Path("results/Untangle/untangle_tamp50_v2"),  # waiting on new results from  Freya
        Path("results/Untangle/untangle_always_blacklist_v1"),  # waiting on new results from Nova
    ]))

    print_results_table(load_data([
        Path("results/Threading/threading_ours_v4"),
        Path("results/Threading/threading_wang_v2"),
        Path("results/Threading/threading_tamp5_v1"),
    ]))

    print_results_table(load_data([
        Path("results/Pulling/pulling_ours_v2"),
    ]))


def load_data(trials_dirs):
    # First find the full list of headers
    headers = []
    for trials_dir in trials_dirs:
        if not trials_dir.exists():
            print(f"WARNING: {trials_dir} does not exist")
            continue
        for trial_dir in trials_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            # load each json file and make a row with the data
            json_path = list(trial_dir.glob("*.json"))[0]
            with json_path.open("r") as f:
                data = json.load(f)
            if len(headers) == 0:
                headers = list(data.keys())
            else:
                # add new headers to the list
                for key in data.keys():
                    if key not in headers:
                        headers.append(key)
    rows = []
    for trials_dir in trials_dirs:
        if not trials_dir.exists():
            print(f"WARNING: {trials_dir} does not exist")
            continue
        for trial_dir in trials_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            # load each json file and make a row with the data
            json_path = list(trial_dir.glob("*.json"))[0]
            with json_path.open("r") as f:
                data = json.load(f)
            # add data in the order of headers
            row = [trials_dir.name]
            for header in headers:
                row.append(data.get(header, np.nan))
            rows.append(row)
    headers.insert(0, 'dirname')
    df = pd.DataFrame(rows, columns=headers)
    return df


def print_results_table(df):
    if len(df) == 0:
        print("No data.")
        return

    agg = {
        'success':        ['sum', 'count'],
        'overall_time':   ['mean', 'std'],
        'sim_time':       ['mean', 'std'],
        'planning_times': list_mean,
        'mpc_times':      list_mean,
        'grasp_history':  [avg_grasps, std_grasps],
    }
    table_data = df.groupby(['dirname']).agg(agg)

    # make a latex table out of the summary data above
    print()
    print("Method & Success & Wall Time (m) & Sim Time (m) & Grasps \\\\")
    for dirname, row in table_data.iterrows():
        x = [
            f"{dirname}",
            f"{row['success']['sum']:.0f}/{row['success']['count']:.0f}",
            f"{row['overall_time']['mean'] / 60:.0f} ({row['overall_time']['std'] / 60:.0f})",
            f"{row['sim_time']['mean'] / 60:.1f} ({row['sim_time']['std'] / 60:.1f})",
            f"{row['grasp_history']['avg_grasps']:.1f} ({row['grasp_history']['std_grasps']:.1f})",
        ]
        print(' & '.join(x) + " \\\\")


if __name__ == '__main__':
    main()
