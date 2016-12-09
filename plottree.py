import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error


def getdummyvars(encodedf, df, cname, fill_na):
    encodedf[cname] = df[cname]
    if fill_na is not None:
        encodedf[cname].fillna(fill_na, inplace=True)
    dummies = pd.get_dummies(encodedf[cname], prefix="_"+cname)
    encodedf = encodedf.join(dummies)
    encodedf = encodedf.drop([cname], axis=1)
    return encodedf


def makedummies(df):
    tempdf = pd.DataFrame(index=df.index)
    tempdf = getdummyvars(tempdf, df, "MSSubClass", None)
    tempdf = getdummyvars(tempdf, df, "MSZoning", None)
    tempdf = getdummyvars(tempdf, df, "LotConfig", None)
    tempdf = getdummyvars(tempdf, df, "Neighborhood", None)
    tempdf = getdummyvars(tempdf, df, "Condition1", None)
    tempdf = getdummyvars(tempdf, df, "Condition2", None)
    tempdf = getdummyvars(tempdf, df, "BldgType", None)
    tempdf = getdummyvars(tempdf, df, "HouseStyle", None)
    tempdf = getdummyvars(tempdf, df, "RoofStyle", None)
    tempdf = getdummyvars(tempdf, df, "RoofMatl", None)
    tempdf = getdummyvars(tempdf, df, "Heating", None)
    tempdf = getdummyvars(tempdf, df, "Exterior1st", "VinylSd")
    tempdf = getdummyvars(tempdf, df, "Exterior2nd", "VinylSd")
    tempdf = getdummyvars(tempdf, df, "Foundation", None)
    tempdf = getdummyvars(tempdf, df, "SaleType", "WD")
    tempdf = getdummyvars(tempdf, df, "SaleCondition", "Normal")
    tempdf = getdummyvars(tempdf, df, "LotShape", None)
    tempdf = getdummyvars(tempdf, df, "LandContour", None)
    tempdf = getdummyvars(tempdf, df, "LandSlope", None)
    tempdf = getdummyvars(tempdf, df, "Electrical", "SBrkr")
    tempdf = getdummyvars(tempdf, df, "GarageType", "None")
    tempdf = getdummyvars(tempdf, df, "GarageQual", "None")
    tempdf = getdummyvars(tempdf, df, "GarageCond", "None")
    tempdf = getdummyvars(tempdf, df, "PoolQC", "None")
    tempdf = getdummyvars(tempdf, df, "PavedDrive", None)
    tempdf = getdummyvars(tempdf, df, "MiscFeature", "None")
    tempdf = getdummyvars(tempdf, df, "Fence", "None")
    tempdf = getdummyvars(tempdf, df, "MoSold", None)
    tempdf = getdummyvars(tempdf, df, "GarageFinish", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtExposure", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtFinType1", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtFinType2", "None")
    tempdf = getdummyvars(tempdf, df, "Functional", "None")
    tempdf = getdummyvars(tempdf, df, "ExterQual", "None")
    tempdf = getdummyvars(tempdf, df, "ExterCond", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtQual", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtCond", "None")
    # By including street and alley encodings:
    # XGBoost score on training set:  0.048088620072
    # Lasso score on training set: 0.101160413666
    tempdf = getdummyvars(tempdf, df, "Street", None)
    tempdf = getdummyvars(tempdf, df, "Alley", None)
    # tempdf = getdummyvars(tempdf, df, "GarageYrBlt", None)
    # tempdf = getdummyvars(tempdf, df, "YearBuilt", None)
    idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
    tempdf.loc[idx, "MasVnrType"] = "BrkFace"
    tempdf = getdummyvars(tempdf, df, "MasVnrType", "None")
    return tempdf


def factorize(tdf, df, column, fill_na=None):
    le = LabelEncoder()
    df[column] = tdf[column]
    if fill_na is not None:
        df[column].fillna(fill_na, inplace=True)
    # else:
    #    df[column].fillna("None", inplace=True)
    le.fit(df[column].unique())
    df[column] = le.transform(df[column])
    return df


def process_data(df, neighborhood_map, bldg_type_map, zone_type_map, qual_map):
    processed_df = pd.DataFrame(index=df.index)
    # For LotFrontage nulls, replacing them with median of lot based on neighbourhood instead of calculating square feet
    lotfn = df["LotFrontage"].groupby(df["Neighborhood"])
    for key, group in lotfn:
        idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())
        processed_df.loc[idx, "LotFrontage"] = group.median()
    processed_df["Age"] = 2010 - df["YearBuilt"]

    processed_df["MoSold"] = df["MoSold"]

    processed_df["NeighborhoodIndicator"] = df["Neighborhood"].map(neighborhood_map)

    processed_df["TotalBsmtSF"] = df["TotalBsmtSF"]
    processed_df["NewHome"] = df["MSSubClass"].replace({20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
                                                        90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
    # New area features:
    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']
    processed_df["*TotalArea"] = df[area_cols].sum(axis=1)
    processed_df["*FloorArea"] = df["1stFlrSF"] + df["2ndFlrSF"]
    processed_df["*TotBathrooms"] = df["BsmtFullBath"] + (df["BsmtHalfBath"]) + df["FullBath"] + (
                                       df["HalfBath"])

    processed_df["OverallQual"] = df["OverallQual"]
    processed_df["OverallCond"] = df["OverallCond"]
    # binning data as new features:
    processed_df["*OverallQual"] = df["OverallQual"].map({1: 1, 2: 1, 3: 1,
                                                          4: 2, 5: 2, 6: 2,
                                                          7: 3, 8: 3, 9: 3, 10: 3})
    # adding a high season feature based on simple hist plot of houses sold
    processed_df["HighlySeasonal"] = df["MoSold"].replace(
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    # Converting any spare nulls to 0s:
    for col in processed_df.columns.values:
        processed_df[col].fillna(0, inplace=True)
    return processed_df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def xgbr(train, train_df):
    # XGB:
    # print(train.head(1))
    # print(trainlabel.head(1))
    numfeats = train.dtypes[train.dtypes != "object"].index
    X = np.array(train[numfeats])
    # X = X[:, 0:80]
    print(X[0])
    y = np.array(train_df["SalePrice"])
    print(y)

    model = xgb.XGBRegressor()
    model.fit(X, y)
    # print(X[0])
    # print(y[0])
    # plot single tree
    plt.style.use("ggplot")
    # Uncomment following to get feature importance by F-score or a DTree
    # plot_tree(model)
    # plot_importance(model)
    plt.show()

    return 0

if __name__ == "__main__":

    train_df = pd.read_csv("train_orig.csv")
    neighborhood_map = {
        "MeadowV": 0,  # 88000
        "IDOTRR": 1,  # 103000
        "BrDale": 1,  # 106000
        "OldTown": 1,  # 119000
        "Edwards": 1,  # 119500
        "BrkSide": 1,  # 124300
        "Sawyer": 1,  # 135000
        "Blueste": 1,  # 137500
        "SWISU": 2,  # 139500
        "NAmes": 2,  # 140000
        "NPkVill": 2,  # 146000
        "Mitchel": 2,  # 153500
        "SawyerW": 2,  # 179900
        "Gilbert": 2,  # 181000
        "NWAmes": 2,  # 182900
        "Blmngtn": 2,  # 191000
        "CollgCr": 2,  # 197200
        "ClearCr": 3,  # 200250
        "Crawfor": 3,  # 200624
        "Veenker": 3,  # 218000
        "Somerst": 3,  # 225500
        "Timber": 3,  # 228475
        "StoneBr": 4,  # 278000
        "NoRidge": 4,  # 290000
        "NridgHt": 4,  # 315000
    }

    bldg_type_map = {
        '2fmCon': 1,  # 127500
        'Duplex': 2,  # 135980
        'Twnhs': 2,  # 137500
        '1Fam': 3,  # 167900
        'TwnhsE': 3  # 172200
    }

    zone_type_map = {
        'A': 1,  # Agriculture
        'C': 2,  # Commercial
        'FV': 3,  # Floating Village Residential
        'I': 4,  # Industrial
        'RH': 5,  # Residential High Density
        'RL': 6,  # Residential Low Density
        'RP': 7,  # Residential Low Density Park
        'RM': 8  # Residential Medium Density
    }
    qual_map = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    proc_train_df = process_data(train_df, neighborhood_map, bldg_type_map, zone_type_map, qual_map)
    xgb_pred = xgbr(proc_train_df, train_df)
