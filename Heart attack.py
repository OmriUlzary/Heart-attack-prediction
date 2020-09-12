#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.datasets import make_regression


def read_data(csv_file):
    try:
        return pd.read_csv(csv_file)
    except:
        print("The file is not found")
        return None


Heart_attack_data_set = read_data("C:/Users/omri1/PycharmProjects/untitled2/heart_attack.csv")


# In[108]:


Heart_attack_data_set


# In[109]:


# Statistical analysis 

Heart_attack_data_set.describe()


# In[110]:


Heart_attack_data_set['sex'].value_counts().plot(kind="bar", title="Sex Variable Distribution", alpha=0.5)
plt.show()


# In[111]:


def data_shape(data, label):
    print('Rows number of ' + label + " is: ", data.shape[0])
    print('Columns number of ' + label + ' is: ', data.shape[1])

def data_columns(data):
    return list(data.columns)

def describe_data(data):
    return data.describe()

data_shape(Heart_attack_data_set, 'Heart attack data set')
data_columns(Heart_attack_data_set)
describe_data(Heart_attack_data_set)


# In[112]:


sns.set(style="whitegrid", palette="muted")
new_data = (Heart_attack_data_set - Heart_attack_data_set.mean()) / (Heart_attack_data_set.std()) 
new_data = pd.concat([Heart_attack_data_set['DEATH_EVENT'], new_data.iloc[:,0:12]], axis=1)
new_data


# In[113]:


new_data = pd.melt(new_data, id_vars="DEATH_EVENT", var_name="features", value_name='value')
new_data


# In[114]:


# 0 = Death
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="DEATH_EVENT", data=new_data)
plt.xticks(rotation=90)


# In[115]:


fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(Heart_attack_data_set.corr(), annot = True, ax=ax)
plt.title('Relations between columns', fontsize = 20)
plt.show()


# In[116]:


drop_list = ['time', 'DEATH_EVENT']
fs_corr = Heart_attack_data_set.drop(columns=drop_list)
fs_corr.head()


# In[117]:


y = Heart_attack_data_set['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(fs_corr, y, test_size=0.2,random_state=42)
clf_rf = RandomForestClassifier(n_estimators=20)
clr_rf = clf_rf.fit(x_train,y_train)


# In[118]:


print('Accuracy',accuracy_score(y_test,clf_rf.predict(x_test)))
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")


# In[119]:


K = range(1, len(x_train.columns))
for k in K:
    select_feature = SelectKBest(chi2, k=k).fit(x_train, y_train)
    scores = zip(x_train.columns, select_feature.scores_)
    print("Selected K:", k)
    for i, (column, score) in enumerate(scores):
        if i < k:
            print("Feature:", column, ", Score:", score)
    print("-----------")


# In[120]:


clf_rf_ = RandomForestClassifier(n_estimators=20)
rfe = RFE(estimator=clf_rf_, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)


# In[121]:


print('Chosen best 5 feature by RFE:',x_train.columns[rfe.support_])


# In[122]:


clf_rf_ = RandomForestClassifier(n_estimators=20)
clr_rf_ = clf_rf_.fit(x_train,y_train)
importances = clr_rf_.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]


# In[123]:


print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[124]:


plt.figure(1, figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[125]:


#SVM algorithm
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
df_feat = Heart_attack_data_set
df_feat.info()
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(y), test_size=0.30, random_state=101)


# In[126]:


model = SVC()
model.fit(X_train,y_train)


# In[127]:


predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))


# In[128]:


print(classification_report(y_test,predictions))


# In[129]:


#Gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[130]:


grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

