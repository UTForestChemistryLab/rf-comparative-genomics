#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import os

#%%
'''
function
    rf: Perform RF.
    bootstrap_rf: Repeat RF on a randomly divided dataset.
    smote_rf: Repeat RF on a randomly divided and over-sampled dataset.
'''
def rf(x_train, x_test, y_train, y_test):
    # x: Number of each CAZymes family, y: ass90_10 (A/G specialist / Generalist)
    # random forest
    clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=777)
    clf.fit(x_train, y_train)
    # feature importance
    max_imp = max(clf.feature_importances_)
    norm = lambda z: z / max_imp
    imp = norm(clf.feature_importances_)
    # prediction
    pred = pd.DataFrame(y_test)
    y_pred = clf.predict(x_test)
    pred["Prediction"] = y_pred
    prob = clf.predict_proba(x_test)
    pred["Angio_Probability"] = prob[:, 0]
    pred["Gen_Probability"] = prob[:, 2]
    pred["Gymno_Probability"] = prob[:, 1]

    return(imp, pred)


def bootstrap_rf(x, y, seed, n_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_trials = pd.DataFrame(index=imp_idx)
    met_trials = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F1-score"])
    a_pred_trials = pd.DataFrame(index=y.index)
    gen_pred_trials = pd.DataFrame(index=y.index)
    g_pred_trials = pd.DataFrame(index=y.index)

    for i in range(n_trials):
        # split data for training and test.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed+i, shuffle=True)
        # rf
        imp, pred = rf(x_train, x_test, y_train, y_test)

        # feature importance
        imp_trials["trial_{}".format(i+1)] = imp
        # metrics
        accuracy = accuracy_score(pred["ass90_10"], pred["Prediction"])
        recall = recall_score(pred["ass90_10"], pred["Prediction"], average="macro")
        precision = precision_score(pred["ass90_10"], pred["Prediction"], average="macro")
        f = f1_score(pred["ass90_10"], pred["Prediction"], average="macro")
        met_trials["trial_{}".format(i+1)] = [accuracy, recall, precision, f]
        # prediction
        pred.drop(["ass90_10", "Prediction"], axis=1, inplace=True)
        pred.rename(columns={
            "Angio_Probability": "trial_{}_A".format(i+1),
            "Gen_Probability": "trial_{}_Gen".format(i+1),
            "Gymno_Probability": "trial_{}_G".format(i+1)
            }, inplace=True)
        a_pred_trials = pd.merge(a_pred_trials, pred["trial_{}_A".format(i+1)], how="left", left_index=True, right_index=True)
        gen_pred_trials = pd.merge(gen_pred_trials, pred["trial_{}_Gen".format(i+1)], how="left", left_index=True, right_index=True)
        g_pred_trials = pd.merge(g_pred_trials, pred["trial_{}_G".format(i+1)], how="left", left_index=True, right_index=True)

    res = [imp_trials, met_trials, a_pred_trials, gen_pred_trials, g_pred_trials]

    for r in res:
        mean = r.mean(axis=1)
        var = r.var(axis=1)
        r.insert(loc=0, column="mean", value=mean)
        r.insert(loc=1, column="var", value=var)

    pred_mean = pd.DataFrame(index=y.index)
    pred_mean["ass90_10"] = y
    pred_mean["A_mean"] = res[2]["mean"]
    pred_mean["A_var"] = res[2]["var"]
    pred_mean["Gen_mean"] = res[3]["mean"]
    pred_mean["Gen_var"] = res[3]["var"]
    pred_mean["G_mean"] = res[4]["mean"]
    pred_mean["G_var"] = res[4]["var"]

    res.append(pred_mean)

    return(res)


def smote_rf(x, y, smote_trials=100, rf_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_df = pd.DataFrame(index=imp_idx)
    met_df = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F1-score"])
    a_pred_df = pd.DataFrame(index=y.index)
    gen_pred_df = pd.DataFrame(index=y.index)
    g_pred_df = pd.DataFrame(index=y.index)
    dfs = [imp_df, met_df, a_pred_df, gen_pred_df, g_pred_df]

    a = y[y==0].count()
    g = y[y==1].count()
    gen = y[y==2].count()
    n_samples = max([a, g, gen])
    sm_strategy = {0: n_samples, 1: n_samples, 2: n_samples}

    for l in range(smote_trials):
        # over-sampling (SMOTE)
        smote = SMOTE(random_state=l*10, sampling_strategy=sm_strategy)
        x_resmp, y_resmp = smote.fit_resample(x, y)
        # reassign index
        sp_idx = y.index.tolist()
        for s in range(len(y_resmp) - len(y)):
            sp_idx.append(f"synthetic_sample_{s+1}")
        x_resmp.index = sp_idx
        y_resmp.index = sp_idx
        # bootstrap cycle for a over-sampled dataset
        bs_cycle = bootstrap_rf(x_resmp, y_resmp, l, n_trials=rf_trials)

        # store result
        del bs_cycle[-1]
        for m, (df, res) in enumerate(zip(dfs, bs_cycle)):
            res.drop(["mean", "var"], axis=1, inplace=True)
            cols = []
            for n in range(len(res.columns)):
                col = "trial_{}".format(l * 10 + n + 1)
                cols.append(col)
            res.columns = cols
            dfs[m] = pd.merge(df, res, how="left", left_index=True, right_index=True)

    for m, df in enumerate(dfs):
        mean = df.mean(axis=1)
        var = df.var(axis=1)
        df.insert(loc=0, column="mean", value=mean)
        df.insert(loc=1, column="var", value=var)

    pred_mean_df = pd.DataFrame(index=y.index)
    pred_mean_df["ass90_10"] = y
    pred_mean_df["A_mean"] = dfs[2]["mean"]
    pred_mean_df["A_var"] = dfs[2]["var"]
    pred_mean_df["Gen_mean"] = dfs[3]["mean"]
    pred_mean_df["Gen_var"] = dfs[3]["var"]
    pred_mean_df["G_mean"] = dfs[4]["mean"]
    pred_mean_df["G_var"] = dfs[4]["var"]

    dfs.append(pred_mean_df)

    return(dfs)



##########################################################
#%%
'''
Preparation for analysis.
ASS: Dataset of CAZymes gene counts for Gymnosperm specialist / Angiosperm specialist / Generalist
["ass90_10"] = Angiosperm specialist: 0, Gymnosperm specialist: 1, Generalist: 2
'''
DF = pd.read_csv("../data/host_association.csv", header=0)
DF.set_index("Project_code", inplace=True)

# All but white/brown-rot data are excluded.
ASS = DF.dropna(subset=["ass90_10"])

# Column "ass90_10" is replaced from str to int.
host_map = {"ass90_10": {"A": 0, "G": 1, "Gen":2}}
ASS.replace(host_map, inplace=True)

# Remove class categories from explanatory variables.
cols = ASS.loc[:, "CAZymes":].columns.tolist()
classes = ["CAZymes", "AA", "CBM", "CE", "GH", "GT", "PL"]
for c in classes:
    cols.remove(c)

##########################################################
#%%
'''
Ex.1
RF on original dataset.
'''
x = ASS.loc[:, cols].astype("int")
y = ASS["ass90_10"]
result_1 = bootstrap_rf(x, y, 0, n_trials=100)
os.makedirs("imbalanced_RF", exist_ok=True)
result_1[0].to_csv("imbalanced_RF/imb_RF_importance.csv")
result_1[1].to_csv("imbalanced_RF/imb_RF_metrics.csv")

rev_map = {"ass90_10": {0:"A", 1:"G", 2:"Gen"}}
result_1[5].replace(rev_map, inplace=True)
result_1[5].to_csv("imbalanced_RF/imb_RF_prediction.csv")
result_1[2].to_csv("imbalanced_RF/imb_RF_pred_A.csv")
result_1[3].to_csv("imbalanced_RF/imb_RF_pred_Gen.csv")
result_1[4].to_csv("imbalanced_RF/imb_RF_pred_G.csv")

##########################################################
#%%
'''
Ex.2
RF on over-sampled dataset.
'''
x = ASS.loc[:, cols].astype("int")
y = ASS["ass90_10"]
result_2 = smote_rf(x, y, smote_trials=100, rf_trials=10)
os.makedirs("smote_RF", exist_ok=True)
result_2[0].to_csv("smote_RF/smote_RF_importance.csv")
result_2[1].to_csv("smote_RF/smote_RF_metrics.csv")

rev_map = {"ass90_10": {0:"A", 1:"G", 2:"Gen"}}
result_2[5].replace(rev_map, inplace=True)
result_2[5].to_csv("smote_RF/smote_RF_prediction.csv")
result_2[2].to_csv("smote_RF/smote_RF_pred_A.csv")
result_2[3].to_csv("smote_RF/smote_RF_pred_Gen.csv")
result_2[4].to_csv("smote_RF/smote_RF_pred_G.csv")



# %%
