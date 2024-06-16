import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#data cleaning

df=pd.read_csv('D:/GUVI/Rock vs Mine Prediction/sonar data.csv',header=None)
df.replace({"M":0,"R":1},inplace=True)

#test,train the data

x=df.drop(60,axis=1)
y=df[60]
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.10,stratify=y,random_state=1)

#Algorithm defining 
model=RandomForestClassifier(criterion="entropy").fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)

# model check through input data
check=np.array([0.1313,0.2339,0.3059,0.4264,0.4010,0.1791,0.1853,0.0055,0.1929,0.2231,0.2907,0.2259,0.3136,0.3302,0.3660,0.3956,0.4386,0.4670,0.5255,0.3735,0.2243,0.1973,0.4337,0.6532,0.5070,0.2796,0.4163,0.5950,0.5242,0.4178,0.3714,0.2375,0.0863,0.1437,0.2896,0.4577,0.3725,0.3372,0.3803,0.4181,0.3603,0.2711,0.1653,0.1951,0.2811,0.2246,0.1921,0.1500,0.0665,0.0193,0.0156,0.0362,0.0210,0.0154,0.0180,0.0013,0.0106,0.0127,0.0178,0.0231])
re_check=check.reshape(1,-1)

y_check=model.predict(re_check)
y_check