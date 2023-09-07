import json

import numpy as np
import pandas as pd
from pathlib import Path


def list_mean(x):
    return np.mean([np.mean(l) for l in x])


def count_grasps(x):
    return np.mean([len(x_i) for x_i in x])


def main():
    untangle_trials_dirs = [
        Path("results/Untangle/untangle_ours_v3"),
        Path("results/Untangle/untangle_tamp5_v2"),
        Path("results/Untangle/untangle_tamp50_v1"),  # waiting on new results from  Freya
    ]

    df = load_data(untangle_trials_dirs)
    print_results_table(df)

    threading_trials_dirs = [
        Path("results/Threading/threading_ours_v2"),
        Path("results/Threading/threading_wang1"),
    ]

    df = load_data(threading_trials_dirs)
    print_results_table(df)


def load_data(trials_dirs):
    # First find the full list of headers
    headers = []
    for trials_dir in trials_dirs:
        for trial_dir in trials_dir.iterdir():
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
        for trial_dir in trials_dir.iterdir():
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
    agg = {
        'success':        ['sum', 'count'],
        'overall_time':   ['mean', 'std'],
        'sim_time':       ['mean', 'std'],
        'planning_times': list_mean,
        'mpc_times':      list_mean,
        'grasp_history':  count_grasps,
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
            f"{row['grasp_history']['count_grasps']:.1f}",
        ]
        print(' & '.join(x) + " \\\\")


if __name__ == '__main__':
    main()
