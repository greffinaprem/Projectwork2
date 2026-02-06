**SAMPLE CODING**



Dataset.csv

&nbsp;file\_writes,process\_count,cpu\_usage,memory\_usage,file\_entropy,label

120,45,78,65,7.9,1

110,40,75,60,7.6,1

95,38,72,58,7.4,1

130,50,82,70,8.1,1

105,42,76,63,7.7,1

30,10,20,25,3.2,0

25,8,18,22,3.0,0

35,12,24,28,3.5,0

40,15,30,35,3.8,0

28,9,19,24,3.1,0



&nbsp;   **data\_preprocessing.py**

import pandas as pd

from sklearn.preprocessing import StandardScaler

import joblib



def load\_dataset(path):

&nbsp;   data = pd.read\_csv(path)

&nbsp;   return data



def preprocess\_data(data):

&nbsp;   X = data.drop("label", axis=1)

&nbsp;   y = data\["label"]



&nbsp;   scaler = StandardScaler()

&nbsp;   X\_scaled = scaler.fit\_transform(X)



&nbsp;   joblib.dump(scaler, "scaler.pkl")

&nbsp;   return X\_scaled, y



**feature\_analysis.py**

import pandas as pd

import matplotlib.pyplot as plt



def analyze\_features(data):

&nbsp;   correlations = data.corr()\["label"].sort\_values(ascending=False)

&nbsp;   print("Feature Correlation with Target:\\n")

&nbsp;   print(correlations)



def plot\_feature\_distribution(data):

&nbsp;   data.hist(figsize=(12, 8))

&nbsp;   plt.tight\_layout()

&nbsp;   plt.show()



**train\_models.py**



import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model\_selection import train\_test\_split

from sklearn.metrics import accuracy\_score

from data\_preprocessing import load\_dataset, preprocess\_data



data = load\_dataset("dataset.csv")

X, y = preprocess\_data(data)



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&nbsp;   X, y, test\_size=0.2, random\_state=42

)



rf = RandomForestClassifier(n\_estimators=150, random\_state=42)

rf.fit(X\_train, y\_train)



svm = SVC(kernel="rbf", probability=True)

svm.fit(X\_train, y\_train)



rf\_acc = accuracy\_score(y\_test, rf.predict(X\_test))

svm\_acc = accuracy\_score(y\_test, svm.predict(X\_test))



print("Random Forest Accuracy:", rf\_acc)

print("SVM Accuracy:", svm\_acc)



joblib.dump(rf, "rf\_model.pkl")

joblib.dump(svm, "svm\_model.pkl")



**evaluate\_models.py**



from sklearn.metrics import classification\_report, confusion\_matrix

import joblib



def evaluate(model, X\_test, y\_test):

&nbsp;   predictions = model.predict(X\_test)

&nbsp;   print(confusion\_matrix(y\_test, predictions))

&nbsp;   print(classification\_report(y\_test, predictions))



**predict\_ransomware.py**



import joblib

import numpy as np



rf = joblib.load("rf\_model.pkl")

svm = joblib.load("svm\_model.pkl")

scaler = joblib.load("scaler.pkl")



def predict(sample):

&nbsp;   scaled = scaler.transform(sample)

&nbsp;   rf\_pred = rf.predict(scaled)\[0]

&nbsp;   svm\_pred = svm.predict(scaled)\[0]



&nbsp;   if rf\_pred == 1 or svm\_pred == 1:

&nbsp;       return "ALERT: RANSOMWARE DETECTED"

&nbsp;   else:

&nbsp;       return "System is Safe"



sample\_input = np.array(\[\[95, 35, 72, 60, 7.3]])

result = predict(sample\_input)

print(result)



**utils.py** 



def log\_message(message):

&nbsp;   with open("system\_log.txt", "a") as file:

&nbsp;       file.write(message + "\\n")



**requirements.txt**

pandas

numpy

scikit-learn

matplotlib

joblib

