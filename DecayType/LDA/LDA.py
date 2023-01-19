#%%
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import os

#%%
##########################################################
'''
function
    corr_ratio: Calculating correlation ratio.
    reduce_families: Reducing families based on correlation ratio and coef.
    lda: Perform LDA.
    bootstrap_lda: Repeat LDA on a randomly divided dataset.
    smote_lda: Repeat LDA on a randomly divided and over-sampled dataset.
'''
def corr_ratio(d):
    l = []
    y = d["Decay_type"]
    for i in range(d.columns.get_loc("CAZymes"), len(d.columns)):
        x = d.iloc[:, i]
        var = ((x - x.mean()) ** 2).sum()
        inter_class = sum([((x[y == j] - x[y == j].mean()) ** 2).sum() for j in np.unique(y)])
        if var != 0:
            rat = 1 - (inter_class / var)
        else:
            rat = 0
        l.append(rat)
    ratio = pd.DataFrame(
        data= l, 
        index= d.columns.values[d.columns.get_loc("CAZymes"):], 
        columns= ["corr_ratio"]
        )
    return(ratio)


def reduce_families(data, corr_r_TH, corr_c_TH):
    # correlation ratio of decay-type and CAZy-families.
    ratio = corr_ratio(data)
    ratio_ext = ratio[ratio["corr_ratio"] > corr_r_TH]
    # correlation coefficient of families.
    corr_c = data.loc[:,ratio_ext.index.values].corr()
    # Families are reduced so that the coef. were below a threshold for all pairs.
    col_1 = ["selected", "rejected", "corr_coef"]
    dropping_table = pd.DataFrame(columns=col_1)
    for i in range(len(ratio_ext)):
        for j in range(len(ratio_ext)):
            if i != j:
                if corr_c_TH < corr_c.iat[i,j]:
                    I, J = ratio_ext.index.values[i], ratio_ext.index.values[j]
                    if ratio_ext.iat[i,0] < ratio_ext.iat[j,0]:
                        pair = pd.DataFrame(data=[[J, I, corr_c.iat[i,j]]], columns=col_1)
                    elif ratio_ext.iat[i,0] > ratio_ext.iat[j,0]:
                        pair = pd.DataFrame(data=[[I, J, corr_c.iat[i,j]]], columns=col_1)
                    else:
                        k, l = max(i, j), min(i, j)
                        K, L = ratio_ext.index.values[k], ratio_ext.index.values[l]
                        pair = pd.DataFrame(data=[[K, L, corr_c.iat[k,l]]], columns=col_1)
                    dropping_table = dropping_table.append(pair, ignore_index=True)
    dropping_table = dropping_table.drop_duplicates()

    for family in ratio_ext.sort_values("corr_ratio", ascending=False).index.values:
        if family in dropping_table["rejected"].values:
            dropping_table = dropping_table[dropping_table["selected"] != family]

    dropping_table.sort_values("selected", inplace=True)
    dropping_table.reset_index(drop=True, inplace=True)
    core_enzymes = ratio_ext.drop(index = dropping_table["rejected"].tolist())

    return(core_enzymes, dropping_table)


def lda(x_train, x_test, y_train, y_test):
    # x: Number of each CAZymes family, y: Decay type (White/Brown-rot)
    sc = StandardScaler()
    x_train_ = sc.fit_transform(x_train)
    x_test_ = sc.fit_transform(x_test)
    # LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(x_train_, y_train).transform(x_train_)

    # Constant
    cv = np.mean(np.dot(lda.means_, lda.scalings_))

    # determine the direction of discriminant function
    if abs(min(lda.scalings_.flatten())) > max(lda.scalings_.flatten()):
        coefs = [c*(-1) for c in lda.scalings_.flatten()]
        cv = cv*(-1)
    else:
        coefs = lda.scalings_.flatten()

    # feature importance
    max_imp = max(coefs)
    norm = lambda z: z / max_imp
    imp = np.append(norm(coefs), norm(cv))

    # prediction
    pred = pd.DataFrame(columns = ["Decay_type", "Prediction", "LDA"])
    for i, species in enumerate(x_test.index.values):
        z = -cv
        for j in range(len(coefs)):
            z = z + x_test_[i,j] * imp[j]
            if z < 0:
                binary = 0
            else:
                binary = 1
            p = np.append([y_test[species], binary], z)
            pred.loc[species] = p


    return(imp, pred)


def bootstrap_lda(x, y, seed, n_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_idx.extend(["constant"])
    imp_trials = pd.DataFrame(index=imp_idx)
    met_trials = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F1-score"])
    pred_trials = pd.DataFrame(index=y.index)

    for k in range(n_trials):
        # split data for training and test.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed+k, shuffle=True)
        # lda
        imp, pred = lda(x_train, x_test, y_train, y_test)

        #feature importance
        imp_trials["trial_{}".format(k+1)] = imp
        # metrics (positive example: Brown-rot, negative example: White-rot)
        accuracy = accuracy_score(pred["Decay_type"], pred["Prediction"])
        recall = recall_score(pred["Decay_type"], pred["Prediction"], pos_label=0)
        precision = precision_score(pred["Decay_type"], pred["Prediction"], pos_label=0)
        f = f1_score(pred["Decay_type"], pred["Prediction"], pos_label=0)
        met_trials["trial_{}".format(k+1)] = [accuracy, recall, precision, f]
        # prediction
        pred.drop(["Decay_type", "Prediction"], axis=1, inplace=True)
        pred.rename(columns={"LDA": "trial_{}".format(k+1)}, inplace=True)
        pred_trials = pd.merge(pred_trials, pred, how="left", left_index=True, right_index=True)

    res = [imp_trials, met_trials, pred_trials]

    for r in res:
        mean = r.mean(axis=1)
        var = r.var(axis=1)
        r.insert(loc=0, column="mean", value=mean)
        r.insert(loc=1, column="var", value=var)

    res[2].insert(loc=0, column="Decay_type", value=y)

    return(res)


def smote_lda(x, y, smote_trials=100, lda_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_idx.extend(["constant"])
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
        bs_cycle = bootstrap_lda(x_resmp, y_resmp, l, n_trials=lda_trials)

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

# Families are reduced based on correlation ratio and correlation coefficient.
core_families, dropping_table = reduce_families(data=WB, corr_r_TH=0.1, corr_c_TH=0.55)
core_families.to_csv("Families_for_Analysis.csv")
dropping_table.to_csv("Correlated_Families.csv", index=False)


##########################################################
#%%
'''
Ex.1
LDA on original dataset.
'''
x = WB.loc[:, core_families.index].astype("int")
y = WB["Decay_type"]
result_1 = bootstrap_lda(x, y, 0, n_trials=100)
os.makedirs("imbalanced_LDA", exist_ok=True)
result_1[0].to_csv("imbalanced_LDA/imb_LDA_importance.csv")
result_1[1].to_csv("imbalanced_LDA/imb_LDA_metrics.csv")

rev_map = {"Decay_type": {0:"Brown-rot", 1:"White-rot"}}
result_1[2].replace(rev_map, inplace=True)
result_1[2].to_csv("imbalanced_LDA/imb_LDA_prediction.csv")


##########################################################
#%%
'''
Ex.2
LDa on over-sampled dataset.
'''
x = WB.loc[:, core_families.index].astype("int")
y = WB["Decay_type"]
result_2 = smote_lda(x, y, smote_trials=100, lda_trials=10)
os.makedirs("smote_LDA", exist_ok=True)
result_2[0].to_csv("smote_LDA/smote_LDA_importance.csv")
result_2[1].to_csv("smote_LDA/smote_LDA_metrics.csv")

rev_map = {"Decay_type": {0:"Brown-rot", 1:"White-rot"}}
result_2[2].replace(rev_map, inplace=True)
result_2[2].to_csv("smote_LDA/smote_LDA_prediction.csv")




# %%
