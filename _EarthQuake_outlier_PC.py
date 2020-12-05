from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import accuracy_score



#EarthQuake
#from google.colab import drive
#drive.mount('/content/drive',force_remount=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sweetviz as sw

#!pip install sweetviz
#import sweetviz as sw
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def outlierStatistic(X_train_predict):
    print(X_train_predict)
    maxX=len(X_train_predict)
    outlier=0
    for i in X_train_predict:
        if i==-1:
            outlier+=1
    print(f"A összes ({maxX} darabból {outlier} darab outlier van. Az {outlier/maxX*100:5.1f} %.)")

basedir="C:/Users/sipocz/OneDrive/Dokumentumok/GitHub"
df=pd.read_csv(basedir+"/_EarthQuake/features_a.csv")
df_clasters=pd.read_csv(basedir+"/_EarthQuake/train_labels.csv")
df_testvalues=pd.read_csv(basedir+"/_EarthQuake/test_a.csv")
#official_testvalues=pd.read_csv("/content/drive/My Drive/001_AI/_EarthQuake/test_values.csv")

# Adatfeldolgozás, beolvasások 

numx=260601
#numx=50000
Y=df_clasters[["damage_grade"]]
X=df
scaler=MinMaxScaler()
Xt=X[:numx]
Xt=scaler.fit_transform(Xt)
Xt=Xt[:numx]
Y=Y[["damage_grade"]]
Y=Y[:numx]
X_train, X_test, y_train, y_test = train_test_split(Xt, Y, random_state=2,test_size=0.0001)
df_buildings=df_testvalues[["building_id"]]
heads=df.columns
print(len(heads))
print(len(df_testvalues.columns))
for i in df_testvalues.columns:
    if i not in heads:
        df_testvalues.drop(columns=[i], inplace=True)

print(len(df_testvalues.columns))
print(len(X_train))
print(X_train)

# inlier és outlier bombázás
# -------------------- Az ötlet -----------------------
# Szedjük a halmazt kétfele 
# inlierekre és outlierekre 
# kezeljük őket külön
# a külön halmazokra illesszünk egy Classifiert 
# OSztályozzuk mindkét halmazt a saját klasszifier modellje szerint
# Fűzzük egybe az adatokat  
#
#

clf = IsolationForest(n_estimators=117, warm_start=False, max_features=45)
clf.fit(X_train)  # fit 10 trees  
#clf.set_params()  # add 10 more trees  
#clf.fit(X)  # fit the added trees 

X_train_predict=clf.predict(X_train)
print(len(X_train))
print(X_train)


outlierStatistic(X_train_predict)

# az official test vizsgálata outlierekre nézve:
outlier_official_predikt=clf.predict(df_testvalues)

outlierStatistic(outlier_official_predikt)

print(list(y_train[["damage_grade"]].index)[0])

print(X_train[1])


y_train["damage_grade"].values


# Szétszedjük a rendszer inlierekre és outlierekre !
X_train_inliers=[X_train[inx] for inx,i in enumerate(X_train_predict) if i==1 ]
X_train_outliers=[X_train[inx] for inx,i in enumerate(X_train_predict) if i==-1 ]
tmp=list(y_train["damage_grade"].values)
y_train_inliers=[tmp[inx] for inx,i in enumerate(X_train_predict) if i==1 ]
y_train_outliers=[tmp[inx] for inx,i in enumerate(X_train_predict) if i==-1 ]



# ---------------- Inlier modell Start ----------------
# konvertáljuk DataFrame - mé a listákat.
dX1=pd.DataFrame (X_train_inliers)
dY1=pd.DataFrame (y_train_inliers)
DtX=pd.DataFrame(X_test)

# KNN inlier modell

from xgboost import XGBClassifier  # 72.09
# max_depth=10 : 72.79857561664441

knn_inlier = XGBClassifier(verbosity=3,max_depth = 35)

print("Fit: Inlier betanitás Start--")
knn_inlier.fit(dX1, dY1)
print("Fit End - Prediction Start ")

y_pred=knn_inlier.predict(DtX)
print("Prediction End")



# Inlier statistics
accuracy =  accuracy_score(y_test, y_pred) * 100
print(f"Accuracy a testhalmazra nézve: {accuracy}")
y_pred2=knn_inlier.predict(dX1)
accuracy =  accuracy_score(y_pred2, y_train_inliers) * 100
print(f"Accuracy a betanított halmazra : {accuracy}")

# ------------- Inlier model vége ---------------

#-------------- Outlier MOdel Start --------------
# DataFrame készítése 
dX1_ol=pd.DataFrame (X_train_outliers)
dY1_ol=pd.DataFrame (y_train_outliers)
DtX=pd.DataFrame(X_test)

# készítsünk modell az outlierekre is
# KNN outlier modell

from xgboost import XGBClassifier  
knn_outlier = XGBClassifier(verbosity=3,max_depth = 35)

print("Fit Start--")
knn_outlier.fit(dX1_ol, dY1_ol)
print("Fit End--")

from sklearn.metrics import accuracy_score
y_pred_ol=knn_outlier.predict(dX1_ol)
accuracy =  accuracy_score(y_pred_ol, y_train_outliers) * 100
print(f"Accuracy a betanított halmazra : {accuracy}")



# megvan a két modell, inlierekre és outlierekre


#az eredeti adatok itt vannak
# ---------------------------------------------------------------------
df_testvalues  #teszteljuk vissza a rendszert  egy eredeti halmazzal 
#----------------------------------------------------------------------
# pl így kapjuk szét az eredeti adatokat
# -- demo adatok esetén --
#X_train, X_test, y_train, y_test = train_test_split(Xt, Y, random_state=2,test_size=0.5)
#df_testvalues=X_test
#----------------------------------------------------------------------
# szét kellene kapni inlierekre és outlierekre a korábban betanított modellel.
test_outliers=clf.predict(df_testvalues)

# nézzünk egy statisztikát
outlierStatistic(test_outliers)

#szedjük szét inlier és outlier listákra


X_test_inliers=[df_testvalues[inx] for inx,i in enumerate(test_outliers) if i==1 ]
X_test_outliers=[df_testvalues[inx] for inx,i in enumerate(test_outliers) if i==-1 ]

df_X_test_inliers=pd.DataFrame (X_test_inliers)
df_X_test_outliers=pd.DataFrame (X_test_outliers)

#predikáljuk az értékeket
#van betanított inlier predikátorunk
#meg van egy az outlierekre is 

y_inlier_predikt=knn_inlier.predict(df_X_test_inliers)
y_outlier_predikt=knn_outlier.predict(df_X_test_outliers)

df_test_buildings=pd.read_csv(basedir+"/_EarthQuake/test_a.csv")
df_buildings=df_test_buildings[["building_id"]]

building_id=df_buildings[["building_id"]].values

print("building_id hossza:",len(building_id))
print("test_outliers hossza:",len(test_outliers))
building=[]
damage=[]
opi=0 # outlier predikt index
ipi=0 # inlier predikt index
bid=0
#össze kellene rakni az eredeti listát
for idx,i in enumerate(test_outliers):
    building.append(building_id[idx][0])  # idx-el kell hivatkozni 
  # a buildin egyelőre nem érdekel, nem végleges adtokkal foglalkozunk !!!
    if i==-1:
        damage.append(y_outlier_predikt[opi])
        opi+=1
    else:
        damage.append(y_inlier_predikt[ipi])    
        ipi+=1
# -- demo adatok esetén --
#accuracy =  accuracy_score(damage, y_test) * 100

print("-----------###------------------------")
print(f"Accuracy a demóra halmazra : {accuracy}")


print(len(building))



outdf=pd.DataFrame(data={"damage_grade":damage} ,index=building)
outdf.index.name="building_id"


outdf.head()

basedir="C:/Users/sipocz/OneDrive/Dokumentumok/GitHub"
outdf.to_csv(basedir+"/_EarthQuake/submission_6_outlier.csv")

#!head "/content/drive/My Drive/001_AI/_EarthQuake/submission_6_outlier.csv"

#!head "/content/drive/My Drive/001_AI/_EarthQuake/test_a.csv"

