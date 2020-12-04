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
# Google
#basedir="/content/drive/My Drive/001_AI/"
#
# Local
basedir="C:/Users/sipocz/OneDrive/Dokumentumok/GitHub/"

df=pd.read_csv(basedir+"_EarthQuake/features_a.csv")
df_clasters=pd.read_csv(basedir+"_EarthQuake/train_labels.csv")
df_testvalues=pd.read_csv(basedir+"_EarthQuake/test_a.csv")
print(df.index)
numx=260601
print("Minta:", numx)
Y=df_clasters[["damage_grade"]]
X=df
scaler=MinMaxScaler()
Xt=X[:numx]
Xt=scaler.fit_transform(Xt)
Xt=Xt[:numx]
Y=Y[["damage_grade"]]
Y=Y[:numx]
X_train, X_test, y_train, y_test = train_test_split(Xt, Y, random_state=0)

heads=df.columns
print(len(heads))
print(len(df_testvalues.columns))
for i in df_testvalues.columns:
    if i not in heads:
        df_testvalues.drop(columns=[i], inplace=True)

print(len(df_testvalues.columns))

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#knn=SVC()

from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=7)

from sklearn.neural_network import MLPClassifier
#knn=MLPClassifier()

from sklearn.gaussian_process import GaussianProcessClassifier
#knn = GaussianProcessClassifier()

from xgboost import XGBClassifier  # 72.09
# max_depth=10 : 72.79857561664441



#knn = XGBClassifier(verbosity=1,max_depth = 10,process_type="default", predictor="gpu_predictor", feature_selector ="shuffle",booster="gbtree")

from sklearn.naive_bayes import GaussianNB

knn = GaussianNB()


print("Fit Start--")
knn.fit(X_train, y_train)

print("Fit End--")

print("Fit Start--")
#knn.fit(X_train, y_train)

print("Fit End--")

y_pred=knn.predict(X_test)
print("Prediction End--")

accuracy =  accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy}")


df_testvalues=scaler.fit_transform(df_testvalues)
Ypredikt=knn.predict(df_testvalues)
#Ypredikt=[]


df_testo=pd.read_csv(basedir+"/_EarthQuake/test_a.csv")



df_testBuilding=df_testo[["building_id"]]


df_out=pd.DataFrame(data=Ypredikt,columns=["damage_grade"], index=list(df_testBuilding["building_id"]),)
df_out.index.name="building_id"
df_out.head()
df_out.to_csv(basedir+"/_EarthQuake/submission_3_GNB.csv")
print("End of Running")













