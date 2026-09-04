import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import roc_curve
import os


files = ["CNN", "GT", "GNN"]

def get_df(path):
    df = pd.read_csv(path + "/eval_metrics/bootstrap_evaluation_metrics.csv")
    return df

def get_rocs(dir):

    path = dir + "/separate"

    rocs = []
    for i in [0]:#range(int(len(os.listdir(path))/2)):
        y_pred_proba = np.load(path + f"/test_scores_{i}.npy").T
        y_test = np.load(path + f"/truth_{i}.npy").T.flatten()

        if np.shape(y_pred_proba) != np.shape(y_test):
            print(np.shape(y_pred_proba), np.shape(y_test))

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
        rocs.append([fpr, tpr, thresholds])
        #print(len(fpr))


    df = pd.DataFrame(rocs, columns = ["fpr", "tpr", "tresholds"])

    
    return df

def ft_score(AUCs, plot = True):
    mdls = ["CNN", "GT", "GNN"]
    ts = [14987, 12840, 1194]

    

    t_min = np.min(ts)
    t_max = np.max(ts)

    font = {'size': 20}
                
    # using rc function
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 2)



    plt.figure(figsize=(6, 4))

    for t_max in [np.max(ts), 20000, 40000, 60000]:
        Ts, FTs = [], []



    # t_min = 0
    # t_max = 20000

        print(f"t_min = {t_min}, t_max = {t_max}")

        t_array = [t_max, t_max, t_max]


        for t in ts:
            T = 1 - (t - t_min) / (t_max - t_min)

            # if T == 0 :
            #     T+=0.01

            Ts.append(T)

        for i in range(len(AUCs)):
            FT = ((1+3**2)*Ts[i]*AUCs[i]/((3**2)*Ts[i] + AUCs[i]))

            FTs.append(FT)

        print(FTs)

        if plot:
            

            #map = mpl.colormaps["viridis"].resampled(8)
            map = ["r", "b", "g"]
            for i in range(len(AUCs)):
                if t_max == np.max(ts):
                    
                    plt.scatter( FTs[i], t_array[i],c = map[i], label = f"{mdls[i]}", s =50, alpha = 1)
                else:
                    plt.scatter( FTs[i], t_array[i],c = map[i], s =50, alpha = 1)
            
    #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.legend()
    plt.xlabel("FT score")
    plt.ylabel("t_max")
    #plt.yscale("log")
    plt.tight_layout()
    plt.grid()
    #plt.savefig("rn325/my_analysis/mlbd_dt_models/AUC_vs_T.png")

    plt.savefig("rn325/my_analysis/mlbd_dt_models/AUC_vs_T.pdf", format="pdf") 
    plt.close()

        # if plot:

        #     # creating a dictionary
        #     font = {'size': 20}

        #     # using rc function
        #     plt.rc('font', **font)
        #     plt.rc('lines', linewidth = 2)


        #     plt.figure(figsize=(6, 4))

        #     map = mpl.colormaps["viridis"].resampled(8)
        #     map = ["r", "b", "g"]
        #     for i in range(len(AUCs)):
        #         plt.scatter(AUCs[i], Ts[i], c = map[i], label = f"{mdls[i]}, FT = {FTs[i]:.2f}", s =50, alpha = 1)
            
        #     #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        #     plt.legend()
        #     plt.xlabel("AUC score")
        #     plt.ylabel("T score")
        #     plt.tight_layout()
        #     plt.grid()
        #     #plt.savefig("rn325/my_analysis/mlbd_dt_models/AUC_vs_T.png")

        #     plt.savefig("rn325/my_analysis/mlbd_dt_models/AUC_vs_T.pdf", format="pdf") 
        #     plt.close()

    return AUCs, Ts, FTs


def plot_all(dfs):

    # creating a dictionary
    font = {'size': 20}

    # using rc function
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 2)


    plt.figure(figsize=(10, 6))
    metric_list = [[], [], []]
    for i in range(len(dfs)):
        df=dfs[i]
        print(df.head())
        these = [1, 2, 3, 4, 6, 7]
        for j in these:
            key= df.keys()[j]
            metric_list[i].append(df[key].to_numpy()[0])
    new = np.array(metric_list)
    # print(np.shape(new))
    print(new)

    # print(y)

    y = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    colour = ["r", "g", "b"]
    lbl = ["CNN", "GT", "GNN"]


    for i in range(3):
        plt.scatter(y[0], new[i]*100, c = colour[i], label = lbl[i])


    plt.grid()
    plt.title("Evaluation metrics of the different ML models")
    plt.ylabel("Percentage")
    plt.xticks([1, 2, 3, 4, 5, 6], ["A", "R", "P", "S", "F1", "AUC"])
    plt.xlabel("Evaluation Metric")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("rn325/wtf.png")
    plt.close()


# creating a dictionary
font = {'size': 20}

# using rc function
plt.rc('font', **font)
plt.rc('lines', linewidth = 2)


plt.figure(figsize=(10, 6))

AUCs = []
frame_list = []

for model in files:

    file = "rn325/my_analysis/mlbd_dt_models/this_model/" + model

    df = get_df(file)

    frame_list.append(df)

    auc = df["auc_score"].to_numpy()[0]

    AUCs.append(auc)

    roc_df = get_rocs(file)

    fpr = roc_df["fpr"][0]
    tpr = roc_df["tpr"][0]

    plt.plot(fpr, tpr, label=f'ROC curve {model}')

# plot_all(frame_list)

# plt.rcParams.update({'font.size': 30})

ft = ft_score(AUCs)

print(ft)

# plt.plot([0, 1], [0, 1], 'k--', label='random guess')
# plt.xlim([-0.05, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for the models')
# plt.legend()
# plt.grid()
# plt.savefig("rn325/my_analysis/mlbd_dt_models/this_model/roc_curve.png")
