#! /usr/bin/env python3

import argparse
import glob
import logging
import os
import sys
import types

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import pandas.plotting
import seaborn as sns
# from scipy.signal import savgol_filter
from collections import defaultdict
from pathlib import Path
from numpy.polynomial.polynomial import Polynomial

__version__ = "0.0.1"

"""
things

"""


def load_file_TT(filepath):
    log = logging.getLogger(__name__)
    log.debug("in")

    df = None
    if ".csv" in filepath:
        df = pd.read_csv(filepath,  skiprows=1)
        df.drop(0, axis=0, inplace=True)
    print(filepath)
    df["Force"] = df["Force"].astype(float)
    df["Stroke"] = df["Stroke"].astype(float)

    # get size and run number and crimp type
    subset_range = len(df) // 2  # Integer division to get half the length

    # Calculate minimum values from the first half of each column
    min_force = df['Force'][:subset_range].min()
    min_stroke = df['Stroke'][:subset_range].min()

    # Normalize the entire column by subtracting the calculated minimums
    df['Force'] = df['Force'] - min_force
    df['Stroke'] = df['Stroke'] - min_stroke

    fname = os.path.basename(filepath)
    slugs = Path(fname).stem
    slugs = slugs.split("-")
    run_num = slugs[-1].split(".")[0]
    if "/even" in filepath:
        C1_L = 32
        C2_L = 32
        offset = int(fname.split("-")[0])
    elif "/uneven" in filepath:
        print(filepath)
        C1_L = int(slugs[0])
        C2_L = int(slugs[1])
        offset = (C1_L - 26)
    elif "/Y-axis" in filepath:
        C1_L = 32
        C2_L = 32
        offset = int(fname.split("-")[0])
    else:
        print("not a know length")
        sys.exit()

    log.info(run_num)
    df.meta = types.SimpleNamespace()
    df.meta.filepath = filepath
    df.meta.test_run = run_num
    df.meta.c1l = C1_L
    df.meta.c2l = C2_L
    df.meta.offset = offset

    log.debug("out")
    return df


def load_file_multi(data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in multi")

    trans_trial = []
    trans_df = []
    for filename in data_paths:
        print(filename)
        df = load_file_TT(filename)
        test_trial = df.meta.offset
        # test_trial = f"{df.meta.c1l}_{df.meta.c2l}"
        trans_trial.append(test_trial)
        trans_df.append(df)

    log.debug("out multi")
    return trans_df, trans_trial


def plot_data_single_trace(param_y="T", param_x="T", data_paths=None):
    """
    plot one data trace from a file

    simple plotting example
    only load data from first file matched in data_paths
    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Time": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Force": {'label': "Force", 'hi': 26, 'lo': 0, 'lbl': "Force (N)"},
        "Stroke": {'label': "Stroke", 'hi': 310,
                   'lo': 0, 'lbl': "Stroke (mm)"},
        "Stress": {'label': "Stress", 'hi': 50,
                   'lo': 0, 'lbl': "Stress (MPa)"},
        "Strain": {'label': "Strain", 'hi': 0.15,
                   'lo': 0, 'lbl': "Strain (mm/mm)"},
    }

    df = load_file_TT(glob.glob(data_paths[0])[0])
    test_trial = df.meta.trial
    test_run_num = df.meta.test_run
    x = df[param_x]
    y = df[param_y]

    figsize = 4  
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):
        line = "{}-vs-{}".format(param_y, param_x)
        ax.plot(x, y, label=line)

        xh = plot_param_options[param_x]['hi']
        xl = plot_param_options[param_x]['lo']
        ax.set_xlim([xl, xh])
        ax.set_xlabel(plot_param_options[param_x]['lbl'])

        yh = plot_param_options[param_y]['hi']
        yl = plot_param_options[param_y]['lo']
        ax.set_ylim([yl, yh])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(),
                  frameon=True, loc='upper left')

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-trial-{}-single-run-{}-{}-vs-{}.png".format(
            test_trial, test_run_num, param_y, param_x)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_data_multi_trace(param_y="T", param_x="T", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Time": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Force": {'label': "Force", 'hi': 26, 'lo': 0, 'lbl': "Force (N)"},
        "Stroke": {'label': "Stroke", 'hi': 310,
                   'lo': 0, 'lbl': "Stroke (mm)"},
        "Stress": {'label': "Stress", 'hi': 50,
                   'lo': 0, 'lbl': "Stress (MPa)"},
        "Strain": {'label': "Strain", 'hi': 0.15,
                   'lo': 0, 'lbl': "Strain (mm/mm)"},
    }

    multi_df, trans_trial = load_file_multi(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        max_values = []
        # Set to keep track of unique labels
        plotted_labels = set()
        config_colors = {}
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                  'orange', 'purple', 'brown', 'pink', 'gray']
        color_idx = 0
        for line, conf in zip(multi_df, trans_trial):
            x = line[param_x]
            y = line[param_y]
            test_run_num = conf
            if test_run_num not in config_colors:
                config_colors[test_run_num] = colors[color_idx % len(colors)]
                color_idx += 1
            # Plot the line
            if test_run_num not in plotted_labels:
                ax.plot(x, y, color=config_colors[test_run_num],
                        label=test_run_num)
                plotted_labels.add(test_run_num)
            else:
                ax.plot(x, y, color=config_colors[test_run_num])
            max_values.append(max(y))

        """
        avg_max = np.mean(max_values)
        std_dev = np.std(max_values)
        """

        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        # Create unique legend entries
        handles, labels = ax.get_legend_handles_labels()

        sorted_labels_handles = sorted(zip(labels, handles),
                                       key=lambda x: int(x[0].split('_')[1]))
        sorted_labels, sorted_handles = zip(*sorted_labels_handles)

        # Update the legend with sorted labels
        ax.legend(sorted_handles, sorted_labels,
                  frameon=True, loc='upper left')

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = f"out/plot-trial--multi-{param_x}-vs-{param_y}.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_fitted_with_error_bands(param_y="T", param_x="T", data_paths=None,
                                 polyorder=3, y_threshold=0.8):
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Time": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Force": {'label': "Force", 'hi': 26, 'lo': 0, 'lbl': "Force (N)"},
        "Stroke": {'label': "Stroke", 'hi': 310,
                   'lo': 0, 'lbl': "Stroke (mm)"},
        "Stress": {'label': "Stress", 'hi': 50,
                   'lo': 0, 'lbl': "Stress (MPa)"},
        "Strain": {'label': "Strain", 'hi': 0.15,
                   'lo': 0, 'lbl': "Strain (mm/mm)"},
    }

    multi_df, trans_trial = load_file_multi(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        combined_data = []

        for df, trial in zip(multi_df, trans_trial):
            x = df[param_x]
            y = df[param_y]
            combined_data.append(pd.DataFrame({param_x: x, param_y: y,
                                               'trial': trial}))

        combined_df = pd.concat(combined_data)

        # Define a colormap
        colormap = plt.get_cmap('tab10')

        # Group by trial and param_x to calculate mean and std
        summary_dfs = []
        for idx, (trial, group_df) in enumerate(combined_df.groupby('trial')):
            group_df = group_df.sort_values(param_x)

            # Filter out data points after a specific y value threshold
            group_df = group_df[group_df[param_y] < y_threshold
                                * group_df[param_y].max()]

            # Apply polynomial fit for smoothing
            p = Polynomial.fit(group_df[param_x], group_df[param_y], polyorder)
            fitted_y = p(group_df[param_x])

            # Calculate the mean and std for the fitted values
            group_df['fitted_y'] = fitted_y
            summary_df = group_df.groupby(param_x).agg({
                'fitted_y': ['mean', 'std']})
            summary_df.columns = ['mean', 'std']
            summary_df.reset_index(inplace=True)
            summary_df['trial'] = trial
            summary_dfs.append((summary_df, colormap(idx % colormap.N)))

        for summary_df, color in summary_dfs:
            trial = summary_df['trial'].iloc[0]
            # Print the index of the maximum value in the mean column

            # max_mean_idx = summary_df['mean'].idxmax()

            # Plot the fitted mean line
            ax.plot(summary_df[param_x], summary_df['mean'],
                    label=f'Fitted {trial}', color=color)

            # Plot the error bands
            '''
            ax.fill_between(summary_df[param_x],
                            summary_df['mean'] - summary_df['std'],
                            summary_df['mean'] + summary_df['std'],
                            alpha=0.2, color=color,
                            label=f'Error band {trial}')
            '''

        # Axis labels and legend
        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        ax.legend(frameon=True, loc='upper left')

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = f"out/plot-trial--fitted-{param_x}-vs-{param_y}.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_average_with_error_bands(param_y="T", param_x="T", data_paths=None,
                                  window_length=51, polyorder=3,
                                  drop_threshold=0.1):
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Time": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Force": {'label': "Force", 'hi': 26, 'lo': 0, 'lbl': "Force (N)"},
        "Stroke": {'label': "Stroke", 'hi': 310,
                   'lo': 0, 'lbl': "Stroke (mm)"},
        "Stress": {'label': "Stress", 'hi': 50,
                   'lo': 0, 'lbl': "Stress (MPa)"},
        "Strain": {'label': "Strain", 'hi': 0.15,
                   'lo': 0, 'lbl': "Strain (mm/mm)"},
    }

    multi_df, trans_trial = load_file_multi(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        combined_data = []

        for df, trial in zip(multi_df, trans_trial):
            x = df[param_x]
            y = df[param_y]
            combined_data.append(pd.DataFrame({param_x: x, param_y: y,
                                               'trial': trial}))

        combined_df = pd.concat(combined_data)

        # Define a colormap
        colormap = plt.get_cmap('tab10')

        # Group by trial and param_x to calculate mean and std
        summary_dfs = []
        for idx, (trial, group_df) in enumerate(combined_df.groupby('trial')):
            # Ensure the data is sorted by the x-axis
            group_df = group_df.sort_values(param_x)

            summary_df = group_df.groupby(level=0).agg(
                {param_x: ['mean', 'std'],
                 param_y: ['mean', 'std']})

            summary_df.columns = [f"{col[0]}_{col[1]}" for col in
                                  summary_df.columns]
            # summary_df.columns = ['mean', 'std']
            summary_df.reset_index(inplace=True)
            summary_df['trial'] = trial
            summary_dfs.append((summary_df, colormap(idx % colormap.N)))

        for summary_df, color in summary_dfs:
            trial = summary_df['trial'].iloc[0]
            # Print the index of the maximum value in the mean column
            max_mean_idx = summary_df[f'{param_y}_mean'].idxmax()

            # Plot the smoothed mean line
            ax.plot(summary_df[f'{param_x}_mean'],
                    summary_df[f'{param_y}_mean'],
                    label=f'Average {trial}', color=color)

            # Mark the maximum force point with a red 'x'
            max_force_x = summary_df[f'{param_x}_mean'].iloc[max_mean_idx]
            max_force_y = summary_df[f'{param_y}_mean'].iloc[max_mean_idx]
            ax.scatter(max_force_x, max_force_y, color='red', marker='x')

        # Axis labels and legend
        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        ax.legend(frameon=True, loc='upper left')

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = f"out/plot-trial--average-{param_x}-vs-{param_y}.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_average_maxes(param_y="T", param_x="T", data_paths=None,
                       window_length=51, polyorder=3,
                       drop_threshold=0.1):
    log = logging.getLogger(__name__)
    log.debug("in")

    multi_df, trans_trial = load_file_multi(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        combined_data = []

        for df, trial in zip(multi_df, trans_trial):
            x = df[param_x]
            y = df[param_y]
            combined_data.append(pd.DataFrame({param_x: x, param_y: y,
                                               'trial': trial}))

        combined_df = pd.concat(combined_data)

        # Define a colormap
        colormap = plt.get_cmap('tab10')

        # Group by trial and param_x to calculate mean and std
        summary_dfs = []
        for idx, (trial, group_df) in enumerate(combined_df.groupby('trial')):
            # Ensure the data is sorted by the x-axis
            group_df = group_df.sort_values(param_x)

            # Detect significant drop in y values
            diff_y = group_df[param_y].diff().fillna(0)
            significant_drop_idx = diff_y[diff_y < - drop_threshold *
                                          group_df[param_y].max()].index
            if not significant_drop_idx.empty:
                max_y_idx = significant_drop_idx[0]
                group_df = group_df.loc[:max_y_idx]

            summary_df = group_df.groupby(level=0).agg(
                {param_y: ['mean', 'std']})
            summary_df.columns = ['mean', 'std']
            summary_df.reset_index(inplace=True)
            summary_df['trial'] = trial
            summary_dfs.append((summary_df, colormap(idx % colormap.N)))

        trials = []
        max_forces_y = []
        std_devs = []

        for summary_df, color in summary_dfs:
            trial = summary_df['trial'].iloc[0]
            print(trial)
            # Print the index of the maximum value in the mean column
            max_mean_idx = summary_df['mean'].idxmax()
            print(f'Trial {trial} - Index of max mean: {max_mean_idx}, ' +
                  f'Max mean value: {summary_df["mean"].max()}')

            max_force_y = summary_df['mean'].iloc[max_mean_idx]
            std_dev = summary_df['std'].iloc[max_mean_idx]

            trials.append(trial)
            max_forces_y.append(max_force_y)
            std_devs.append(std_dev)

        ax.errorbar(trials, max_forces_y, yerr=std_devs, fmt='o',
                    linestyle='-', color='blue', label='Max Force Y')

        # Axis labels and legend
        ax.set_xlabel("Displacment (mm)")
        ax.set_ylabel("Force (N)")

        ax.legend(frameon=True, loc='upper left')

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-trial--average-trial_maxf.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def extract_max_values_and_statistics(base_dir, column_name):
    # Get the list of subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_dir) if
               os.path.isdir(os.path.join(base_dir, d))]

    # Dictionary to store maximum values by offset
    max_values_by_offset = defaultdict(list)

    # Iterate over each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        # Get the list of CSV files in the subdirectory
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]

        # Iterate over each CSV file
        force = []
        stroke = []
        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)
            try:
                # Read the CSV file using load_file_TT
                df = load_file_TT(csv_path)
                L1 = float(df.meta.c1l) * 1e-3
                L2 = float(df.meta.c2l) * 1e-3
                L0 = ((L1 + L2) / 2) - float(df.meta.offset) * 1e-3
                print(L0)
                E = 1014 * 1e6
                b = 0.01
                h = 0.002
                I_2nd = (1 / 12) * b * h ** 3
                df["C1"] = ((df["Force"] / 2) * (L0 ** 2)) / (2 * E * I_2nd)
                df["Stroke"] = df["Stroke"] * 1e3
                df["delta_0"] = (df["Stroke"]
                                 - df["Stroke"].min()) / (2 * (L0))
                force.append(df["Force"])
                stroke.append(df["Stroke"])
                # Extract the maximum value from the specified column
                max_value = df[column_name].max()
                # Extract the offset identifier from the metadata
                offset = df.meta.offset
                # Append the maximum value to the appropriate offset group
                max_values_by_offset[offset].append(max_value)
            except Exception as e:
                print(f"Error processing file {csv_path}: {e}")

        plt.figure(figsize=(10, 6))
        for f, s in zip(force, stroke):
            plt.scatter(s,f)
        plt.xlabel("stroke")
        plt.ylabel("force")
        plt.grid(True)
        #plt.show()

    # Array to store mean and standard deviation for each offset group
    statistics = []

    ## Compute mean and standard deviation for each offset group
    #for offset, values in max_values_by_offset.items():
    #    mean_value = np.mean(values)
    #    std_value = np.std(values)
    #    statistics.append((offset, mean_value, std_value))
    #    print(f"Offset: {offset}, Mean: {mean_value}, Std: {std_value}")
    # Compute mean and standard deviation for each offset group and remove outliers
    outlier_threshold =0.75
    for offset, values in max_values_by_offset.items():
        print(values)
        mean_value = np.mean(values)
        std_value = np.std(values)

        # Identify outliers
        non_outliers = [v for v in values if abs(v - mean_value) <= outlier_threshold * std_value]

        # Update the dictionary with non-outlier values
        max_values_by_offset[offset] = non_outliers

        # Recompute mean and std after outliers are removed
        new_mean_value = np.mean(non_outliers)
        new_std_value = np.std(non_outliers)

        statistics.append((offset, new_mean_value, new_std_value))
        print(f"Offset: {offset}, Mean after outlier removal: {new_mean_value}, Std: {new_std_value}")


    statistics.sort(key=lambda x: float(x[0]))

    return statistics




def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    # addl keys: asctime, module, name
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))

    return


def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    # -h/--help is auto-added
    parser.add_argument("-d", "--dir", dest="dirs",
                        # action="store",
                        nargs='+',
                        default=None, required=False,
                        help="directories with data files")
    parser.add_argument("-i", "--in", dest="input",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="path to input")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])

    return ret


def load_data_fits(filepath):
    df = pd.read_hdf(filepath)
    return df


def calc_terms(filepath, theta, L):
    df = load_data_fits(filepath)
    C1p_coefs = df["C1_fit"][theta]
    C4p_coefs = df["C4_fit"][theta]
    dB_lim = df["deltaB_lims"][theta]

    C1_eq = np.poly1d(C1p_coefs)
    C4_eq = np.poly1d(C4p_coefs)

    step = np.mean(dB_lim) / 100
    deltaB = np.arange(dB_lim[0], dB_lim[-1], step)

    C1 = C1_eq(deltaB)
    C4 = C4_eq(deltaB)

    res_C1 = np.empty([len(C1), len(L)])
    res_C4 = np.empty([len(C4), len(L)])
    res_deltaB = np.empty([len(deltaB), len(L)])

    for i in range(len(L)):
        res_C1[:, i] = C1 / L[i]**2
        res_C4[:, i] = C4 * L[i]
        res_deltaB[:, i] = deltaB * L[i]
    return res_C1, res_C4, res_deltaB


def plot_data(param_x="T", param_y="T",  filepath=None, theta=None, L=None):
    """
    """
    log = logging.getLogger(__name__)
    log.debug("in")

    res_C1, res_C4, deltaB = calc_terms(filepath, theta, L)

    plot_param_options = {
        "C1_L":{'label':"$C1/L^2$", 'hi':300, 'lo':0, 'lbl':"$C1 / L^2$"},
        "C4_L":{'label':"C4 $s L$ ", 'hi':300, 'lo':0, 'lbl':"$C4 * L$"},
        "deltaB":{'label':"deltaB * L", 'hi':300, 'lo':0, 'lbl':"$deltaB*L$"},
        "L":{'label':"deltaB$ ", 'hi':300, 'lo':0, 'lbl':"L$"},
    }
    if param_x == "C1_L":
        x = res_C1 #+ flip(res_C1,1)
    elif param_x == "C4_L":
        x = res_C4 #+ flip(res_C4,1)
    elif param_x == "deltaB":
        x = deltaB
    else:
        x = L

    if param_y == "C1_L":
        y = res_C1 #+ flip(res_C1,1)
    elif param_y == "C4_L":
        y = res_C4 #+ flip(res_C4,1)
    elif param_y == "deltaB":
        y = deltaB
    else:
        y = L

    print(np.shape(y))

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        try:
            if len(np.shape(y)) ==2:
                length = range(np.shape(y)[1])
                for i in length:
                    if param_x == "deltaB":
                        xx = x[:,i]
                    else:
                        xx = x[:,i]
                    ax.plot(xx, y[:,i], label="{}L".format(round(L[i],3)))
                    #sys.exit()
        except TypeError:
            ax.plot(x, y)
        ax.legend()
        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot--{}-vs-{}.png".format(param_x, param_y)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

def plot_data_add(param_x="T", param_y="T",  filepath=None, theta=None, L=None):
    """
    """
    log = logging.getLogger(__name__)
    log.debug("in")

    res_C1, res_C4, deltaB = calc_terms(filepath, theta, L)

    plot_param_options = {
        "C1_L":{'label':"$C1/L^2$", 'hi':300, 'lo':0, 'lbl':"$C1 / L^2$"},
        "C4_L":{'label':"C4 $s L$ ", 'hi':300, 'lo':0, 'lbl':"$C4 * L$"},
        "deltaB":{'label':"deltaB * L", 'hi':300, 'lo':0, 'lbl':"$deltaB*L$"},
        "L":{'label':"deltaB$ ", 'hi':300, 'lo':0, 'lbl':"L$"},
    }
    if param_x == "C1_L":
        x = res_C1 #+ flip(res_C1,1)
    elif param_x == "C4_L":
        x = res_C4 #+ flip(res_C4,1)
    elif param_x == "deltaB":
        x = deltaB
    else:
        x = L

    if param_y == "C1_L":
        y = res_C1 #+ flip(res_C1,1)
    elif param_y == "C4_L":
        y = res_C4 #+ flip(res_C4,1)
    elif param_y == "deltaB":
        y = deltaB
    else:
        y = L

    print(np.shape(y))

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        try:
            if len(np.shape(y)) ==2:
                length = range(np.shape(y)[1])

                max_c1 = []
                for i in range(len(L) // 2):
                    idx1 = i  # Index for first element
                    idx2 = len(L) - i - 1  # Index for corresponding element from the end

                    # Plot pair of corresponding elements
                    if param_x == "deltaB":
                        xx1 = x[:, idx1]
                        xx2 = x[:, idx2]
                    else:
                        xx1 = x[:, idx1]
                        xx2 = x[:, idx2]

                    # Plot the pair with appropriate label
                    #ax.plot(xx1 + xx2, y[:, idx1] + y[:, idx2], label="{}L + {}L".format(round(L[idx1], 3), round(L[idx2], 3)))
                    max_c1.append(max(y[:, idx1] + y[:, idx2]))

                    #sys.exit()
                idx_5 = len(L) // 2  # Index for 5
                #ax.plot(x[:, idx_5] * 2, y[:, idx_5] * 2, label="{}L + {}L".format(round(L[idx_5], 3), round(L[idx_5], 3)))

                max_c1.append(max(y[:, idx_5] *2))
                x = np.arange(0.0,0.5,0.05)
                x = np.flip(x)
                print(len(x))
                print(len(max_c1))
                ax.plot(x, max_c1, color='blue', linestyle='-', marker='o',markerfacecolor='red',label="Max Values")
                ax.legend(loc='lower center',ncol=2)
                ax.set_xlabel("Effective Translated Distance")
                #ax.set_xlabel(plot_param_options[param_x]['lbl'])
                ax.set_ylabel(plot_param_options[param_y]['lbl'])
                ax.set_yscale('log')

                logging.info(sys._getframe().f_code.co_name)
                plot_fn = "out/plot--{}-vs-{}.png".format(param_x, param_y)
                logging.info("write plot to: {}".format(plot_fn))
                fig.savefig(plot_fn, bbox_inches='tight')

        except TypeError:
            #ax.plot(x, y)
            print("error")
    return

def calc_components_lr_uneven(filepath, L, R):
    L0 = np.round(L-1e-3,4)
    R0 = np.round(R-1e-3,4)
    d = np.arange(0,0.1,0.001)
    theta = 0
    E = 1014e6
    b = 14.7e-3
    h = 0.6e-3
    I = (1/12)*b*h**3
    deltaB_star_L = d / (2 * L0)
    deltaB_star_R = d / (2 * R0)
    df = load_data_fits(filepath)
    C1p_coefs = df["C1_fit"][theta]
    C4p_coefs = df["C4_fit"][theta]
    C1_eq = np.poly1d(C1p_coefs)
    C4_eq = np.poly1d(C4p_coefs)
    C4_L = C4_eq(deltaB_star_L)
    C4_R = C4_eq(deltaB_star_R)
    C1_L = C1_eq(deltaB_star_L)
    C1_R = C1_eq(deltaB_star_R)

    F_L = (2*E*I*C1_L)/L0**2
    F_R = (2*E*I*C1_R)/R0**2
    P = F_L + F_R
    # Compute the first derivative
    first_derivative = np.diff(P)

    # Compute the second derivative
    second_derivative = np.diff(first_derivative)

    # Find the indices where the second derivative changes sign
    inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]

    if len(inflection_points) > 0:
        first_inflection_index = inflection_points[0] + 2  # +2 to adjust index for the original array

        # Locate the minimum value in C1_L after the inflection point
        min_index_after_inflection = np.argmin(P[first_inflection_index:]) + first_inflection_index

        # Set all values after the minimum index to 0
        P[min_index_after_inflection + 1:] = np.min(P[first_inflection_index:])
        print(f"L:{L}->F_L:{max(F_L[:min_index_after_inflection])} & R:{R}->F_R:{max(F_R[:min_index_after_inflection])}")

    P_max = max(P)

    # plt.figure(figsize=(10, 6))
    # # plt.plot(deltaB_star_L, C1_L, label='C1_L vs. deltaB_star_L', marker='o')
    # # plt.plot(deltaB_star_R, C1_R, label='C1_R vs. deltaB_star_R', marker='o')
    # plt.plot(deltaB_star_R, C4_R, label='d vs. P', marker='o')
    # plt.plot(deltaB_star_L, C4_L, label='d vs. P', marker='o')
    # plt.xlabel('deltaB_star_L')
    # plt.ylabel('C1_L')
    # plt.title('Plot of deltaB_star_L vs. C1_L')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return P_max


def plt_uneven_cantilevers(filepath):
    Ls = np.arange(0.026, 0.041, 0.001).round(4)
    Rs = np.arange(0.012, 0.027, 0.001).round(4)
    Rs = np.flip(Rs)
    p_maxs = []
    for L, R in zip(Ls, Rs):
        p_max = calc_components_lr_uneven(filepath, L, R)
        p_maxs.append(p_max)

    offsets = np.arange(len(p_maxs))

    return p_maxs, offsets


def calc_components_lr_even(filepath, offset):
    L0 = 32e-3 -7e-3 + offset*1e-3
    R0 = 32e-3 -7e-3 - offset*1e-3
    d = np.arange(0,0.1,0.001)
    theta = 0
    E = 1014e6
    b = 14.7e-3
    h = 0.6e-3
    I = (1/12)*b*h**3
    deltaB_star_L = d / (2 * L0)
    deltaB_star_R = d / (2 * R0)
    df = load_data_fits(filepath)
    C1p_coefs = df["C1_fit"][theta]
    C4p_coefs = df["C4_fit"][theta]
    C1_eq = np.poly1d(C1p_coefs)
    C4_eq = np.poly1d(C4p_coefs)
    C1_L = C1_eq(deltaB_star_L)
    C1_R = C1_eq(deltaB_star_R)

    C4_L = C4_eq(deltaB_star_L)
    C4_R = C4_eq(deltaB_star_R)

    F_L = (2*E*I*C1_L)/L0**2
    F_R = (2*E*I*C1_R)/R0**2
    P = F_L + F_R
    # Compute the first derivative
    first_derivative = np.diff(P)

    # Compute the second derivative
    second_derivative = np.diff(first_derivative)

    # Find the indices where the second derivative changes sign
    inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]

    if len(inflection_points) > 0:
        first_inflection_index = inflection_points[0] + 2  # +2 to adjust index for the original array

        # Locate the minimum value in C1_L after the inflection point
        min_index_after_inflection = np.argmin(P[first_inflection_index:]) + first_inflection_index

        # Set all values after the minimum index to 0
        P[min_index_after_inflection + 1:] = np.min(P[first_inflection_index:])

    P_max = max(P)

    return P_max


def plt_even_cantilevers(filepath):
    p_maxs = []
    offsets = np.arange(24)/3
    for i in offsets:
        p_max = calc_components_lr_even(filepath, i)
        p_maxs.append(p_max)

    return p_maxs, offsets


def plot_statistics(path, param):
    """
    Plot the statistics with error bars and show the plot.

    Parameters:
    - statistics:list of tuples, where each tuple contains (offset, mean, std).
    """
    statistics = extract_max_values_and_statistics(path, param)
    offsets, means, stds = zip(*statistics)

    filepath = "data/Fit_Curve_theta-60-60-step-5.h5"
    if "uneven" in path:
        model_data, points = plt_uneven_cantilevers(filepath)
    else:
        model_data, points = plt_even_cantilevers(filepath)

    model_data = np.array(model_data) / model_data[0]
    stds = np.array(stds) / np.abs(means[0])
    stds = stds/ np.sqrt(len(stds))
    means  = np.array(means) / means[0]

    points = np.array(points)/32
    offsets = np.array(offsets)/32

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    ax.plot(points, model_data, linestyle='--', color='green', label="Model")
    ax.errorbar(offsets, means, yerr=stds, fmt='o',
                linestyle='-', color='blue', ecolor='red', capsize=5, label="Exp. Data")
    ax.set_xlabel('Normalized Offset by Cantilever Length in X')
    ax.set_ylabel('Normalized Mean Max. Force')
    ax.set_title('Normalized Mean Max. Force vs Offset in X')
    ax.legend()
    # plt.xticks(rotation=45)
    plt.tight_layout()

    # plt.show()

    save_dir = os.path.basename(os.path.normpath(path))
    out_dir = 'out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, f'{save_dir}-mean-statistics.png')
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")


def calc_components_Y(filepath, offset):
    L0 = np.sqrt((32e-3 -7e-3)**2 + (offset*1e-3)**2)
    d = np.arange(0,0.1,0.001)
    theta = 0
    E = 1014e6
    b = 14.7e-3
    h = 0.6e-3
    I = (1/12)*b*h**3
    deltaB_star = d / (2 * L0)
    df = load_data_fits(filepath)
    C1p_coefs = df["C1_fit"][theta]
    C4p_coefs = df["C4_fit"][theta]
    C1_eq = np.poly1d(C1p_coefs)
    C4_eq = np.poly1d(C4p_coefs)
    C1 = C1_eq(deltaB_star)
    C4 = C4_eq(deltaB_star)
    F = (2*E*I*C1)/L0**2
    P = F*2
    # Compute the first derivative
    first_derivative = np.diff(P)
    # Compute the second derivative
    second_derivative = np.diff(first_derivative)
    # Find the indices where the second derivative changes sign
    inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]
    if len(inflection_points) > 0:
        first_inflection_index = inflection_points[0] + 2  # +2 to adjust index for the original array
        # Locate the minimum value in C1_L after the inflection point
        min_index_after_inflection = np.argmin(P[first_inflection_index:]) + first_inflection_index
        # Set all values after the minimum index to 0
        P[min_index_after_inflection + 1:] = np.min(P[first_inflection_index:])
    P_max = max(P)
    return P_max


def plt_cantilevers_Y(filepath):
    p_maxs = []
    offsets = np.arange(24)/3
    for i in offsets:
        p_max = calc_components_Y(filepath, i)
        p_maxs.append(p_max)
    return p_maxs, offsets


def plot_statisticsY(path, param):
    """
    Plot the statistics with error bars and show the plot.

    Parameters:
    - statistics:list of tuples, where each tuple contains (offset, mean, std).
    """
    statistics = extract_max_values_and_statistics(path, param)
    offsets, means, stds = zip(*statistics)

    filepath = "data/Fit_Curve_theta-60-60-step-5.h5"
    model_data, points = plt_cantilevers_Y(filepath)

    model_data = np.array(model_data) / model_data[0]
    stds = np.array(stds) / np.abs(means[0])
    stds = stds/ np.sqrt(len(stds))
    means = means/max(means)

    points = np.array(points)/32
    offsets = np.array(offsets)/32

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    ax.plot(points, model_data, linestyle='--', color='green', label="Model")
    ax.errorbar(offsets, means, yerr=stds, fmt='o',
                linestyle='-', color='blue', ecolor='red', capsize=5, label="Exp. Data")
    ax.set_xlabel('Normalized Offset by Cantilever Length in Y')
    ax.set_ylabel('Normalized Mean Max. Force')
    ax.set_title('Normalized Mean Max. Force vs Offset in Y')
    ax.legend()
    ax.set_ylim(0.95,)
    ax.set_xlim(-0.01,0.2)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()

    # plt.show()

    save_dir = os.path.basename(os.path.normpath(path))
    out_dir = 'out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, f'{save_dir}-mean-statistics.png')
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")


def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])

    plot_statistics('data/even-trials-low', 'Force')
    plot_statisticsY('data/Y-axis/', 'Force')


if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("exited")
