
#_Dengue

_PCVERSION_=True


if _PCVERSION_:
    basedir="C:/Users/sipocz/OneDrive/Dokumentumok/GitHub"
else:
    from google.colab import drive
    drive.mount('/content/drive',force_remount=True)
    basedir="/content/drive/My Drive/001_AI"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#--------------scikit import 
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#--------------
def outlierStatistic(X_train_predict):
    print(X_train_predict)
    maxX=len(X_train_predict)
    outlier=0
    for i in X_train_predict:
        if i==-1:
            outlier+=1
    print(f"A összes ({maxX} darabból {outlier} darab outlier van. Az {outlier/maxX*100:5.1f} %.)")

features_train=basedir+"/_EarthQuake/train_values.csv"
labels_train=basedir+"/_EarthQuake/train_labels.csv"
features_predict=basedir+"/_EarthQuake/test_values.csv"

X_train=pd.read_csv(features_train)
y_train=pd.read_csv(labels_train)
X_pred=pd.read_csv(features_predict)


def checkvalues(df,columnname,key):
    print(f"{columnname} ellenőrzése !")
    numok=0
    numerr=0
    for i in df.index:
        if df.at[i,columnname] in key:
            #print(df.at[i,columnname])
            numok+=1
            pass
        else:
            numerr+=1
            print(df.at[i,columnname],end=", ")
    sumall=numok+numerr
    print(f"\n{sumall} mintából {numerr} db nem volt megfelelő")


def createcolumn(df,columnname,keys):
    print(f"{columnname} cseréje megy")
    for key in keys:
        df[keys[key]]=0.0
    for key in keys:
        for i in df.index:
            if df.at[i,columnname]==key:
                df.at[i,keys[key]]=1.0


def create_dict(idx,list):
    o={}
    for i in list:
        o[i]=idx+"_"+i
    return o

def create_base_data(df):
    t=['n', 't', 'o']
    columnname="land_surface_condition"
    key=create_dict(columnname,t)


    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t= ['h', 'w', 'i', 'r', 'u']
    columnname="foundation_type"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t=  ['q', 'n', 'x']
    columnname="roof_type"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t=  ['z', 'v', 'f', 'm', 'x']
    columnname="ground_floor_type"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t=   ['q', 's', 'j', 'x']
    columnname="other_floor_type"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t=   ['j', 's', 't', 'o']
    columnname="position"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t=   ['c', 's', 'f', 'd', 'm', 'a', 'q', 'u', 'n', 'o']
    columnname="plan_configuration"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    t=['a', 'w', 'r', 'v']
    columnname="legal_ownership_status"
    key=create_dict(columnname,t)

    #checkvalues(df,columnname,key)
    createcolumn(df,columnname,key)

    df["geopos"]=df.geo_level_1_id+df.geo_level_2_id/10000+df.geo_level_3_id/1000000000
    return(df)

X_pred_conv=create_base_data(X_pred)
X_train_conv=create_base_data(X_train)

def kill_columns(df):
    notkey=["building_id","legal_ownership_status","geo_level_1_id",	"geo_level_2_id",	"geo_level_3_id", "land_surface_condition",	"foundation_type",	"roof_type",	"ground_floor_type",	"other_floor_type",	"position",	"plan_configuration"]
    for i in df.columns:
        if i in notkey:
            df.drop(columns=[i], inplace=True)
    return df


X_train_ok=kill_columns(X_train_conv)
X_pred_ok=kill_columns(X_pred_conv)
y_train_ok=kill_columns(y_train)


from xgboost import XGBClassifier  # 72.09
# max_depth=10 : 72.79857561664441

knn = XGBClassifier(verbosity=3,max_depth = 95, n_estimators=400)

print("Fit: Inlier betanitás Start--")
knn.fit(X_train_ok, y_train_ok)
print("Fit End - Prediction Start ")

y_pred=knn.predict(X_train_ok)
print("Prediction End")

accuracy =  accuracy_score(y_pred, y_train_ok) * 100
print(f"Accuracy a betanított halmazra : {accuracy}")

#outfile generation
y_pred_ok=knn.predict(X_pred_ok)
print("Prediction End")

X_pred_bd=pd.read_csv(features_predict)

buildingid=X_pred_bd["building_id"]
head2=y_pred_ok


outdf=pd.DataFrame(data={"damage_grade":y_pred_ok} ,index=buildingid)
outdf.index.name="building_id"


outdf.head()


outdf.to_csv(basedir+"/_EarthQuake/submission_10_xgboost.csv")
print()
print(basedir+"/_EarthQuake/submission_9_xgboost.csv")