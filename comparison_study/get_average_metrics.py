# "rn325/my_analysis/mlbd_dt_models/DM_GNN/outputs/eval/1296650/bootstrap_evaluation_metrics"

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def average_df(file, verbose = False, save = True, plot_confusion = True):
    df = pd.read_csv(file)
    cl = 0.95

    metrics = list(df.keys())[1:]
    calculated = [[], [], []]

    for metric in metrics:
        metric_array = df[metric].to_numpy()
        average_metric = np.mean(metric_array)


        calculated[0].append(average_metric)

        ci = stats.t.interval(cl, df=len(metric_array)-1, loc=np.mean(metric_array), scale=np.std(metric_array, ddof=1) / np.sqrt(len(metric_array)))

        calculated[1].append(ci[0])
        calculated[2].append(ci[1])

    new_df = pd.DataFrame(calculated, columns = metrics, index = ["average", "lower_bound", "upper_bound"])

    if verbose:
        print(new_df.head())

    if save:
        new_df.to_csv(f"{file}_averaged.csv")

    if plot_confusion:
        print_confusion(new_df, file)

    return 0

def print_confusion(df, path, plot_absolute = False):

    confusion_matrix_arrs = np.array([[df["tn"]["average"], df["fp"]["average"]], [df["fn"]["average"], df["tp"]["average"]]])

    norm_conf_matrix = confusion_matrix_arrs / np.sum(confusion_matrix_arrs, axis=1, keepdims=True)


    # creating a dictionary
    font = {'size': 20}

    # using rc function
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 2)


    plt.figure(figsize=(10, 6))

    if plot_absolute:
        plt.pcolormesh(confusion_matrix_arrs, cmap='Blues', shading='auto')
        plt.xticks([0.5, 1.5], ['Background', 'Signal'])
        plt.yticks([0.5, 1.5], ['Background', 'Signal'])
        for i in range(2):
            for j in range(2):
                if i == j:
                    colour = "white"
                else:
                    colour = "black"
                plt.text(j+0.5, i+0.5, confusion_matrix_arrs[i, j], ha='center', va='center', color=colour, fontsize=20)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.colorbar(label='Counts')
        plt.tight_layout()
        plt.show()
        title = path + "_conf_matrix_absolute.png"
        plt.savefig(title)
        plt.close()

    plt.pcolormesh(norm_conf_matrix, cmap='Blues', shading='auto')
    plt.xticks([0.5, 1.5], ['Background', 'Signal'])
    plt.yticks([0.5, 1.5], ['Background', 'Signal'])
    for i in range(2):
        for j in range(2):
            if i == j:
                colour = "white"
            else:
                colour = "black"
            plt.text(j+0.5, i+0.5, f'{norm_conf_matrix[i][j]:.2f}', ha='center', va='center', color=colour, fontsize=20)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.colorbar(label='percentage')
    plt.tight_layout()
    plt.show()

    title = path + "_conf_matrix_perc.png"
    plt.savefig(title)
    plt.close()

    return 0


fl = "rn325/my_analysis/mlbd_dt_models/this_model/GNN/eval_metrics/GNN_bootstrap_evaluation_metrics.csv"

average_df(fl, verbose = True)
