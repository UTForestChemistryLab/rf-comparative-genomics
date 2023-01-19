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
    # x: Number of each CAZymes family, y: Decay type (White/Brown-rot)
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
    pred["Probability"] = prob[:, [True, False]]

    return(imp, pred)


def bootstrap_rf(x, y, seed, n_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_trials = pd.DataFrame(index=imp_idx)
    met_trials = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F1-score"])
    pred_trials = pd.DataFrame(index=y.index)

    for i in range(n_trials):
        # split data for training and test.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed+i, shuffle=True)
        # rf
        imp, pred = rf(x_train, x_test, y_train, y_test)

        # feature importance
        imp_trials["trial_{}".format(i+1)] = imp
        # metrics (positive example: Brown-rot, negative example: White-rot)
        accuracy = accuracy_score(pred["Decay_type"], pred["Prediction"])
        recall = recall_score(pred["Decay_type"], pred["Prediction"], pos_label=0)
        precision = precision_score(pred["Decay_type"], pred["Prediction"], pos_label=0)
        f = f1_score(pred["Decay_type"], pred["Prediction"], pos_label=0)
        met_trials["trial_{}".format(i+1)] = [accuracy, recall, precision, f]
        # prediction
        pred.drop(["Decay_type", "Prediction"], axis=1, inplace=True)
        pred.rename(columns={"Probability": "trial_{}".format(i+1)}, inplace=True)
        pred_trials = pd.merge(pred_trials, pred, how="left", left_index=True, right_index=True)

    res = [imp_trials, met_trials, pred_trials]

    for r in res:
        mean = r.mean(axis=1)
        var = r.var(axis=1)
        r.insert(loc=0, column="mean", value=mean)
        r.insert(loc=1, column="var", value=var)

    res[2].insert(loc=0, column="Decay_type", value=y)

    return(res)


def smote_rf(x, y, smote_trials=100, rf_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_df = pd.DataFrame(index=imp_idx)
    met_df = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F1-score"])
    pred_df = pd.DataFrame(index=y.index)
    dfs = [imp_df, met_df, pred_df]

    for l in range(smote_trials):
        # over-sampling (SMOTE)
        smote = SMOTE(random_state=l*10, sampling_strategy=1)
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
        for m, (df, res) in enumerate(zip(dfs, bs_cycle)):
            if m == 2:
                res.drop(["mean", "var", "Decay_type"], axis=1, inplace=True)
            else:
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

    dfs[2].insert(loc=0, column="Decay_type", value=y)

    return(dfs)




##########################################################
#%%
'''
Preparation for analysis.
WB: Dataset of CAZymes gene counts for White/Brown-rot
["Decay_type"] = Brown-rot: 0, White-rot: 1
'''
DF = pd.read_csv("../data/decay_type.csv", header=0)
DF.set_index("Project_code", inplace=True)

# All but white/brown-rot data are excluded.
WB = DF[(DF["Decay_type"] == "White-rot") | (DF["Decay_type"] == "Brown-rot")]

# Column "Decay_type" is replaced from string to binary.
Decay_type_map = {"Decay_type": {"Brown-rot": 0, "White-rot": 1}}
WB.replace(Decay_type_map, inplace=True)

# Remove class categories from explanatory variables.
cols = WB.loc[:, "CAZymes":].columns.tolist()
classes = ["CAZymes", "AA", "CBM", "CE", "GH", "GT", "PL"]
for c in classes:
    cols.remove(c)

##########################################################
#%%
'''
Ex.1
RF on original dataset.
'''
x = WB.loc[:, cols].astype("int")
y = WB["Decay_type"]
result_1 = bootstrap_rf(x, y, 0, n_trials=100)
os.makedirs("imbalanced_RF", exist_ok=True)
result_1[0].to_csv("imbalanced_RF/imb_RF_importance.csv")
result_1[1].to_csv("imbalanced_RF/imb_RF_metrics.csv")

rev_map = {"Decay_type": {0:"Brown-rot", 1:"White-rot"}}
result_1[2].replace(rev_map, inplace=True)
result_1[2].to_csv("imbalanced_RF/imb_RF_prediction.csv")


##########################################################
#%%
'''
Ex.2
RF on over-sampled dataset.
'''
x = WB.loc[:, cols].astype("int")
y = WB["Decay_type"]
result_2 = smote_rf(x, y, smote_trials=100, rf_trials=10)
os.makedirs("smote_RF", exist_ok=True)
result_2[0].to_csv("smote_RF/smote_RF_importance.csv")
result_2[1].to_csv("smote_RF/smote_RF_metrics.csv")

rev_map = {"Decay_type": {0:"Brown-rot", 1:"White-rot"}}
result_2[2].replace(rev_map, inplace=True)
result_2[2].to_csv("smote_RF/smote_RF_prediction.csv")

# %%
