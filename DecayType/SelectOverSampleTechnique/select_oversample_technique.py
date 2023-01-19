#%%
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler





#%%
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
            z = z + x_test_[i,j] * coefs[j]
            if z < 0:
                binary = 0
            else:
                binary = 1
            p = np.append([y_test[species], binary], z)
            pred.loc[species] = p

    max_pred = max(pred["LDA"].abs())
    norm2 = lambda x: x / max_pred
    pred["LDA"] = pred["LDA"].map(norm2)

    return(imp, pred)


def bootstrap_lda(x, y, seed, n_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_idx.extend(["constant"])
    imp_trials = pd.DataFrame(index=imp_idx)
    met_trials = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F-measure"])
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


def oversmp_lda(x, y, method, os_trials=100, lda_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_idx.extend(["constant"])
    imp_df = pd.DataFrame(index=imp_idx)
    met_df = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F-measure"])
    pred_df = pd.DataFrame(index=y.index)
    dfs = [imp_df, met_df, pred_df]

    for l in range(os_trials):
        # over-sampling (SMOTE)
        if method == "SMOTE":
            model = SMOTE(random_state=l*10, sampling_strategy=1)

        elif method == "Borderline-SMOTE":
            model = BorderlineSMOTE(random_state=l*10, sampling_strategy=1)

        elif method == "ADASYN":
            model = ADASYN(random_state=l*10, sampling_strategy=1)

        elif method == "ROSE":
            model = RandomOverSampler(random_state=l*10, sampling_strategy=1)

        x_resmp, y_resmp = model.fit_resample(x, y)
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
    met_trials = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F-measure"])
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


def oversmp_rf(x, y, method, os_trials=100, rf_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_df = pd.DataFrame(index=imp_idx)
    met_df = pd.DataFrame(index=["Accuracy", "Recall", "Precision", "F-measure"])
    pred_df = pd.DataFrame(index=y.index)
    dfs = [imp_df, met_df, pred_df]

    for l in range(os_trials):
        # over-sampling (SMOTE)
        if method == "SMOTE":
            model = SMOTE(random_state=l*10, sampling_strategy=1)

        elif method == "Borderline-SMOTE":
            model = BorderlineSMOTE(random_state=l*10, sampling_strategy=1)

        elif method == "ADASYN":
            model = ADASYN(random_state=l*10, sampling_strategy=1)

        elif method == "ROSE":
            model = RandomOverSampler(random_state=l*10, sampling_strategy=1)

        x_resmp, y_resmp = model.fit_resample(x, y)
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



#%%
DF = pd.read_csv("../data/decay_type.csv", header=0)
DF.set_index("Project_code", inplace=True)

# All but white/brown-rot data are excluded.
WB = DF[(DF["Decay_type"] == "White-rot") | (DF["Decay_type"] == "Brown-rot")]

# Column "Decay_type" is replaced from string to binary.
Decay_type_map = {"Decay_type": {"Brown-rot": 0, "White-rot": 1}}
WB.replace(Decay_type_map, inplace=True)

# Families are reduced based on correlation ratio and coef.
core_families = pd.read_csv("../LDA/Families_used_for_Analysis.csv", header=0, index_col=0)
x_lda = WB.loc[:, core_families.index].astype("int")

cols = WB.loc[:, "CAZymes":].columns.tolist()
classes = ["CAZymes", "AA", "CBM", "CE", "GH", "GT", "PL"]
for c in classes:
    cols.remove(c)
x_rf = WB.loc[:, cols].astype("int")
y = WB["Decay_type"]


###############################################################
#%%
# imbalanced (not over-sampled)
lda_raw = bootstrap_lda(x_lda, y, 0, n_trials=100)
lda_met = lda_raw[1].drop(["mean", "var"], axis=1)
lda_met.insert(loc=0, column="method", value="imbalanced")
lda_result = pd.DataFrame(lda_met)

rf_raw = bootstrap_rf(x_rf, y, 0, n_trials=100)
rf_met = rf_raw[1].drop(["mean", "var"], axis=1)
rf_met.insert(loc=0, column="method", value="imbalanced")
rf_result = pd.DataFrame(rf_met)


methods = ["SMOTE", "Borderline-SMOTE", "ADASYN", "ROSE"]

for method in methods:
    # lda
    lda_oversmp = oversmp_lda(x_lda, y, method, os_trials=10, lda_trials=10)
    lda_met = lda_oversmp[1].drop(["mean", "var"], axis=1)
    lda_met["method"] = method
    lda_result = pd.concat([lda_result, lda_met], axis=0)
    # rf
    rf_oversmp = oversmp_rf(x_rf, y, method, os_trials=10, rf_trials=10)
    rf_met = rf_oversmp[1].drop(["mean", "var"], axis=1)
    rf_met["method"] = method
    rf_result = pd.concat([rf_result, rf_met], axis=0)

for r in [lda_result, rf_result]:
    mean = r.mean(axis=1)
    var = r.var(axis=1)
    r.insert(loc=1, column="mean", value=mean)
    r.insert(loc=2, column="var", value=var)

lda_result.insert(loc=1, column="metrics", value=lda_result.index.tolist())
rf_result.insert(loc=1, column="metrics", value=rf_result.index.tolist())



###############################################################
#%%
lda_result.to_csv("result_LDA2.csv", index=False)
rf_result.to_csv("result_RF2.csv", index=False)




# %%
