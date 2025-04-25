# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## Developed By: CHARU NETHRA R
## Register no: 212223230035

# CODING AND OUTPUT:
```py
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/703168d0-d1e2-4484-b5fa-46d17029cecc)

```py
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/8d1b2cf1-3f18-4bac-8ce2-ead9ce22498a)

```py
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/271adfea-be93-4e33-9e45-a4e4ac67a379)

```py
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/b277c0cb-00c7-4ea1-8ede-e9f14a0e106a)

```py
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/9ac1bea1-4618-4839-ac1e-fa927d4d1ac9)

```py
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/26b040ab-9a2d-4585-8b3b-8f064383fcc4)

```py
data2
```
![image](https://github.com/user-attachments/assets/7b17ad81-7cc5-4eed-9924-1437272255d8)

```py
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/5e4976fa-aca2-4eea-b65b-7a867b05e7ca)

```py
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/06de2460-32b8-4b31-9529-e0f8058d9447)

```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/7ca4d06c-ae15-4102-bfa9-cc3a3bfa2f49)

```py
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/8b981abc-278a-467d-b5d1-51041f434937)

```py
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/6314acdb-1e9f-4bf9-aadd-f651008ad187)

```py

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/6154f513-25c1-48d3-9525-e6b8a10cfef0)

```py
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/b9bc1c86-2c95-4d19-9647-acef25f68c36)

```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/42e5565f-7f57-4b00-bc3f-b41a822a8373)

```py
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/4c1e51de-d331-4280-88c4-88cb82723074)

```py
data.shape
```
![image](https://github.com/user-attachments/assets/101c06d3-2615-427c-80b1-4da23f5c0cd0)

```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/f494bbdf-deab-47de-85ee-b11d7c126ec2)

```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/4bb9f8b2-2eac-4558-af0a-2994895a4c97)

```py
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/40cd11ea-214e-4a0a-bfd2-56506fe53c93)

```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/400159e3-29db-4ba0-9645-a0632ec63b0e)

```py
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/75de93be-2b73-423a-9551-6efc00465cd9)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
