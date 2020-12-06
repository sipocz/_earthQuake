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
X_train, X_test, y_train, y_test = train_test_split(Xt, Y, random_state=2,test_size=0.5)
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


# neural netrwork 

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-6,verbose=True,max_iter=6000, hidden_layer_sizes=(20,20, 20,20), random_state=1)

clf.fit(X_train, y_train)

y_predict=clf.predict(X_test)
acc=accuracy_score(y_predict,y_test)
print(acc)