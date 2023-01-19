#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import smogn
import os

#%%
'''
function
    rf: Perform RF.
    bootstrap_rf: Repeat RF on a randomly divided dataset.
    smote_rf: Repeat RF on a randomly divided and over-sampled dataset.
'''
def rf(x_train, x_test, y_train, y_test):
    # x: Number of each CAZymes family, y: gymno_freq
    # random forest
    rgs = RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=777)
    rgs.fit(x_train, y_train)
    # feature importance
    max_imp = max(rgs.feature_importances_)
    norm = lambda z: z / max_imp
    imp = norm(rgs.feature_importances_)
    # prediction
    pred = pd.DataFrame(y_test)
    y_pred = rgs.predict(x_test)
    pred["Prediction"] = y_pred

    return(imp, pred)


def bootstrap_rf(x, y, seed, n_trials=10):
    # dataframe to store result
    imp_idx = x.columns.to_list()
    imp_trials = pd.DataFrame(index=imp_idx)
    met_trials = pd.DataFrame(index=["R2", "MAE", "MSE", "RMSE"])
    pred_trials = pd.DataFrame(index=y.index)

    for i in range(n_trials):
        # split data for training and test.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed+i, shuffle=True)
        # rf
        imp, pred = rf(x_train, x_test, y_train, y_test)

        # feature importance
        imp_trials["trial_{}".format(i+1)] = imp
        # metrics (r2, MAE, MSE, RMSE)
        r2 = r2_score(pred["gymno_freq"], pred["Prediction"])
        mae = mean_absolute_error(pred["gymno_freq"], pred["Prediction"])
        mse = mean_squared_error(pred["gymno_freq"], pred["Prediction"], squared=True)
        rmse = mean_squared_error(pred["gymno_freq"], pred["Prediction"], squared=False)
        met_trials["trial_{}".format(i+1)] = [r2, mae, mse, rmse]
        # prediction
        pred.drop(["gymno_freq"], axis=1, inplace=True)
        pred.rename(columns={"Prediction": "trial_{}".format(i+1)}, inplace=True)
        pred_trials = pd.merge(pred_trials, pred, how="left", left_index=True, right_index=True)

    res = [imp_trials, met_trials, pred_trials]

    for r in res:
        mean = r.mean(axis=1)
        var = r.var(axis=1)
        r.insert(loc=0, column="mean", value=mean)
        r.insert(loc=1, column="var", value=var)

    res[2].insert(loc=0, column="gymno_freq", value=y)

    return(res)


def smogn_rf(x, y, data, smogn_trials=100, rf_trials=10):
    # dataframe to store result
    imp_idx = x.columns.to_list()
    imp_df = pd.DataFrame(index=imp_idx)
    met_df = pd.DataFrame(index=["R2", "MAE", "MSE", "RMSE"])
    pred_df = pd.DataFrame(index=y.index)
    dfs = [imp_df, met_df, pred_df]

    xy = pd.concat([y,x], axis=1)
    xy.reset_index(inplace=True, drop=True)

    for l in range(smogn_trials):
        # smogn
        rg_mtrx = [
            [0.5,  1, 0],  ## over-sample ("minority")
            [1, 0, 0],  ## under-sample ("majority")
            [0, 0, 0],  ## under-sample ("majority")
        ]
        for trial in range(0, 10):
            try:
                xy_smogn = smogn.smoter(
                    data=xy,
                    y=y.name,
                    samp_method="extreme",
                    rel_thres=0.3,
                    rel_method="manual",
                    rel_ctrl_pts_rg=rg_mtrx,
                    replace=False
                    )
                break
            except ValueError as e:
                print("error:{e} retry:{t}/10".format(e=e, t=trial))
                continue

        xy_resmp = pd.merge(
            data, xy_smogn,
            how="right",
            left_on=xy.columns.to_list(),
            right_on=xy.columns.to_list()
            )

        # reassign index
        for s in range(xy_resmp["Project_code"].isnull().sum()):
            xy_resmp.loc[s, "Project_code"] = f"synthetic_sample_{s+1}"
        xy_resmp.set_index("Project_code", inplace=True)
        sp_idx = y.index.tolist()
        x_resmp = xy_resmp.loc[:, x.columns].astype("int")
        y_resmp = xy_resmp["gymno_freq"]

        # random forest
        bs_cycle= bootstrap_rf(x_resmp, y_resmp, l, n_trials=rf_trials)

        # store result
        for m, (df, res) in enumerate(zip(dfs, bs_cycle)):
            if m == 2:
                res.drop(["mean", "var", "gymno_freq"], axis=1, inplace=True)
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

    dfs[2].insert(loc=0, column="gymno_freq", value=y)

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
ASS = DF.dropna(subset=["gymno_freq"])

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
y = ASS["gymno_freq"]
result_1 = bootstrap_rf(x, y, 0, n_trials=100)
os.makedirs("imbalanced_RF", exist_ok=True)
result_1[0].to_csv("imbalanced_RF/imb_RF_importance.csv")
result_1[1].to_csv("imbalanced_RF/imb_RF_metrics.csv")
result_1[2].to_csv("imbalanced_RF/imb_RF_prediction.csv")


##########################################################
#%%
'''
RF on over-sampled dataset.
'''
ass_ex2 = ASS.reset_index()
x = ASS.loc[:, cols].astype("int")
y = ASS["gymno_freq"]
result_2 = smogn_rf(x, y, ass_ex2, smogn_trials=100, rf_trials=10)
os.makedirs("smogn_RF", exist_ok=True)
result_2[0].to_csv("smogn_RF/smogn_RF_importance.csv")
result_2[1].to_csv("smogn_RF/smogn_RF_metrics.csv")
result_2[2].to_csv("smogn_RF/smogn_RF_prediction.csv")



# %%
