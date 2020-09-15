#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


Heart_attack_data_set


# In[3]:


# Statistical analysis 

Heart_attack_data_set.describe()


# In[4]:


# From the first analysis I can conclude that most of the ages are on their 60, with a little deviation.
# To most of the examined there no diabetes, blood pressure, or smoking issues.


# In[5]:


Heart_attack_data_set['sex'].value_counts().plot(kind="bar", title="Sex Variable Distribution", alpha=0.5)
plt.show()


# In[6]:


# There are almost two times male tested than female tested.


# In[7]:


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


# In[8]:


Smoking = Heart_attack_data_set[Heart_attack_data_set['smoking'] == 1]['smoking'].sum()
Diabetes = Heart_attack_data_set[Heart_attack_data_set['diabetes'] == 1]['diabetes'].sum()
High_blood_pressure = Heart_attack_data_set[Heart_attack_data_set['high_blood_pressure'] == 1]['high_blood_pressure'].sum()
anaemia = Heart_attack_data_set[Heart_attack_data_set['anaemia'] == 1]['anaemia'].sum()

print('Reasons for heart attack: \nSmoking = ' + str(Smoking) + '\nDiabetes = ' + str(Diabetes) + '\nHigh blood pressure = ' +str(High_blood_pressure)
     + '\nAnemia = ' + str(anaemia))
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

reasons = ['Smoking','Diabetes','High_blood_pressure','Anemia']

data = [Smoking, Diabetes, High_blood_pressure, anaemia]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(reasons[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),horizontalalignment=horizontalalignment, **kw)

ax.set_title('Reasons for heart attack')
plt.show()


# In[9]:


# Most of the heart attacks were caused by anemia but all the numbers are very close to the mentioned reasons.


# In[10]:


Smokers = Heart_attack_data_set[Heart_attack_data_set['smoking'] == 1]['DEATH_EVENT'].sum()
Non_smokers = Heart_attack_data_set[Heart_attack_data_set['smoking'] == 0]['DEATH_EVENT'].sum()

High_blood_pressure = Heart_attack_data_set[Heart_attack_data_set['high_blood_pressure'] == 1]['DEATH_EVENT'].sum()
Non_high_blood_pressure = Heart_attack_data_set[Heart_attack_data_set['high_blood_pressure'] == 0]['DEATH_EVENT'].sum()

Diabetes = Heart_attack_data_set[Heart_attack_data_set['diabetes'] == 1]['DEATH_EVENT'].sum()
Non_diabetes = Heart_attack_data_set[Heart_attack_data_set['diabetes'] == 0]['DEATH_EVENT'].sum()

Deaths = ['Smokers', 'Non-smokers']
slices = [Smokers, Non_smokers]
colors = ['r','b']
plt.pie(slices, labels= Deaths, colors=colors, startangle=90, shadow=True, explode=(0, 0),radius=1.4, autopct='%1.1f%%')
plt.legend()
plt.title('Death by smoking')
plt.show()

Deaths = ['High_blood_pressure', 'Non_high_blood_pressure']
slices = [High_blood_pressure, Non_high_blood_pressure]
colors = ['r','b']
plt.pie(slices, labels= Deaths, colors=colors, startangle=90, shadow=True, explode=(0, 0),radius=1.4, autopct='%1.1f%%')
plt.legend()
plt.title('Death by high blood pressure')
plt.show()

Deaths = ['Diabetes', 'Non_diabetes']
slices = [Diabetes, Non_diabetes]
colors = ['r','b']
plt.pie(slices, labels= Deaths, colors=colors, startangle=90, shadow=True, explode=(0, 0),radius=1.4, autopct='%1.1f%%')
plt.legend()
plt.title('Death by diabetes')
plt.show()


# In[11]:


#Smoking, diabetes and high blood pressure do not necessarily affect death from a heart attack.


# In[12]:


sns.set(style="whitegrid", palette="muted")
new_data = (Heart_attack_data_set - Heart_attack_data_set.mean()) / (Heart_attack_data_set.std()) 
new_data = pd.concat([Heart_attack_data_set['DEATH_EVENT'], new_data.iloc[:,0:12]], axis=1)
new_data


# In[13]:


new_data = pd.melt(new_data, id_vars="DEATH_EVENT", var_name="features", value_name='value')
new_data


# In[14]:


# 0 = Death
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="DEATH_EVENT", data=new_data)
plt.xticks(rotation=90)


# In[15]:


fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(Heart_attack_data_set.corr(), annot = True, ax=ax)
plt.title('Relations between columns', fontsize = 20)
plt.show()


# In[16]:


#There is a high relation between the sex of the individual to smoking but the interesting conclusion is 
#that have a very strong relationship between the serum sodium and age to heart attacks 
#and negative relation between ejection fraction to heart attacks.


# In[17]:


drop_list = ['time', 'DEATH_EVENT']
fs_corr = Heart_attack_data_set.drop(columns=drop_list)
fs_corr.head()


# In[18]:


y = Heart_attack_data_set['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(fs_corr, y, test_size=0.2,random_state=42)
clf_rf = RandomForestClassifier(n_estimators=20)
clr_rf = clf_rf.fit(x_train,y_train)


# In[19]:


print('Accuracy',accuracy_score(y_test,clf_rf.predict(x_test)))
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")


# In[ ]:


#The algorithm has 63.3 success percent in the prediction if the heart attack will be fatal
#according to the personal details of the individual (smoking/non-smoking, sex, age, etc).


# In[20]:


K = range(1, len(x_train.columns))
for k in K:
    select_feature = SelectKBest(chi2, k=k).fit(x_train, y_train)
    scores = zip(x_train.columns, select_feature.scores_)
    print("Selected K:", k)
    for i, (column, score) in enumerate(scores):
        if i < k:
            print("Feature:", column, ", Score:", score)
    print("-----------")


# In[21]:


clf_rf_ = RandomForestClassifier(n_estimators=20)
rfe = RFE(estimator=clf_rf_, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)


# In[22]:


print('Chosen best 5 feature by RFE:',x_train.columns[rfe.support_])


# In[23]:


clf_rf_ = RandomForestClassifier(n_estimators=20)
clr_rf_ = clf_rf_.fit(x_train,y_train)
importances = clr_rf_.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]


# In[24]:


print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[25]:


plt.figure(1, figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[ ]:


#The attribute which affects death from a heart attack most is the level of serum creatinine in the blood.


# In[26]:


#SVM algorithm
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
df_feat = Heart_attack_data_set
df_feat.info()
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(y), test_size=0.30, random_state=101)


# In[27]:


model = SVC()
model.fit(X_train,y_train)


# In[28]:


predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))


# In[29]:


print(classification_report(y_test,predictions))


# In[30]:


#Gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[31]:


grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


# In[ ]:


# The SVM and Grid Search algorithms gave 69 percent for success in the prediction.
# It is a really great prediction in consider to the few amount of data we got from the CSV sheet.


# In[ ]:




