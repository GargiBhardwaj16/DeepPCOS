#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the libraries that I use in this project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTEN
from sklearn.metrics import roc_curve, auc


# In[2]:


import lightgbm as lgb


# In[2]:


pip install lightgbm


# In[3]:


import pandas as pd


# In[3]:


PCOS_inf = pd.read_csv(r'C:\Users\dushy\Downloads\archive (12)\PCOS_infertility.csv')
PCOS_woinf = pd.read_csv(r'C:\Users\dushy\Downloads\PCOS_without_infertility - Full_new.csv')


# In[4]:


data = pd.merge(PCOS_woinf,PCOS_inf, on='Patient File No.', suffixes={'','_y'},how='left')

#Dropping the repeated features after merging
data =data.drop(['Unnamed: 44', 'Sl. No_y', 'PCOS (Y/N)_y', '  I   beta-HCG(mIU/mL)_y',
       'II    beta-HCG(mIU/mL)_y', 'AMH(ng/mL)_y'], axis=1)

#Taking a look at the dataset
data.head() 


# In[5]:


data["AMH(ng/mL)"].head() 


# In[6]:


data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

#Dealing with missing values. 
#Filling NA values with the median of that feature.

data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median(),inplace=True)
data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median(),inplace=True)
data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median(),inplace=True)
data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(),inplace=True)

#Clearing up the extra space in the column names (optional)

data.columns = [col.strip() for col in data.columns]


# In[7]:


corrmat = data.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,cmap="Pastel1", square=True);


# In[9]:


corrmat["PCOS (Y/N)"].sort_values(ascending=False)


# In[8]:


import numpy as np


# In[9]:


plt.figure(figsize=(12,12))
k = 12 #number of variables with positive for heatmap
l = 3 #number of variables with negative for heatmap
cols_p = corrmat.nlargest(k, "PCOS (Y/N)")["PCOS (Y/N)"].index 
cols_n = corrmat.nsmallest(l, "PCOS (Y/N)")["PCOS (Y/N)"].index
cols = cols_p.append(cols_n) 

cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True,cmap="Pastel1", annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[11]:


# Length of menstrual phase in PCOS vs normal 
color = ["teal", "plum"]
fig=sns.lmplot(data=data,x="Age (yrs)",y="Cycle length(days)", hue="PCOS (Y/N)",palette=color)
plt.show(fig)


# In[12]:


# Pattern of weight gain (BMI) over years in PCOS and Normal. 
fig= sns.lmplot(data =data,x="Age (yrs)",y="BMI", hue="PCOS (Y/N)", palette= color )
plt.show(fig)


# In[13]:


# cycle IR wrt age 
sns.lmplot(data =data,x="Age (yrs)",y="Cycle(R/I)", hue="PCOS (Y/N)",palette=color)
plt.show()


# In[14]:


sns.lmplot(data =data,x='Follicle No. (R)',y='Follicle No. (L)', hue="PCOS (Y/N)",palette=color)
plt.show()


# In[15]:


features = ["Follicle No. (L)","Follicle No. (R)"]
for i in features:
    sns.swarmplot(x=data["PCOS (Y/N)"], y=data[i], color="black", alpha=0.5 )
    sns.boxenplot(x=data["PCOS (Y/N)"], y=data[i], palette=color)
    plt.show()


# In[16]:


features = ["Age (yrs)","Weight (Kg)", "BMI", "Hb(g/dl)", "Cycle length(days)","Endometrium (mm)" ]
for i in features:
    sns.swarmplot(x=data["PCOS (Y/N)"], y=data[i], color="black", alpha=0.5 )
    sns.boxenplot(x=data["PCOS (Y/N)"], y=data[i], palette=color)
    plt.show()


# In[17]:


X=data.drop(["PCOS (Y/N)","Sl. No","Patient File No."],axis = 1) #droping out index from features too
y=data["PCOS (Y/N)"]


# In[18]:


print(X.columns)


# In[19]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(35).plot(kind='barh')
plt.rcParams["figure.figsize"] = (30,30)
plt.show()


# In[20]:


fimp = feat_importances.nlargest(9)
df_again = data[fimp.index]
df_again


# In[21]:


map_characters = {0: 'Negative', 1: 'Positive'}
dict_characters=map_characters
import seaborn as sns
plt.subplots(1, figsize=(5,5))
df = pd.DataFrame()
df["labels"]=y
lab = df['labels']
dist = lab.value_counts()

sns.countplot(lab)
plt.savefig('SVM-un-balance.png', bbox_inches='tight')
print(dict_characters)


# In[22]:


from imblearn.over_sampling import ADASYN
from imblearn import under_sampling, over_sampling, combine

ada = over_sampling.ADASYN(sampling_strategy='minority', random_state=None, n_neighbors=3, n_jobs=None)
X_ada, y_ada = ada.fit_resample(df_again, y)

enn = under_sampling.EditedNearestNeighbours(sampling_strategy='all', n_neighbors=3, kind_sel='mode', n_jobs=None)
X_sm, y_sm= enn.fit_resample(X_ada, y_ada)

#sm = SMOTEN(random_state=42)

#X_sm, y_sm = sm.fit_resample(df_again, y)

X_sm=pd.DataFrame(X_sm, 
            columns=df_again.columns)
y_sm=pd.DataFrame(y_sm, columns=['PCOS (Y/N)'])


print('New balance of 1 and 0 classes (%):')
y_sm.value_counts()


# In[23]:


map_characters = {0: 'Negative', 1: 'Positive'}
dict_characters=map_characters
import seaborn as sns
plt.subplots(1, figsize=(1,1))
df = pd.DataFrame()
df["labels"]=y_sm
lab = df['labels']
dist = lab.value_counts()

sns.countplot(lab)
plt.savefig('SVM-Balanced.png', bbox_inches='tight')
print(dict_characters)


# In[24]:


X_train,X_test, y_train, y_test = train_test_split(X_sm,y_sm, test_size=0.2, random_state=12) 


# In[25]:


y_train = np.array(y_train)
y_test = np.array(y_test)


# In[26]:


# import SVC classifier
from sklearn.svm import SVC


# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


# instantiate classifier with linear kernel and C=1.0
linear_svc1000=SVC(kernel='rbf', C=1000.0, probability=True, gamma='scale') 


# fit classifier to training set
linear_svc1000.fit(X_train.values,np.ravel(y_train))


# In[32]:


from deepchecks.tabular.suites import model_evaluation
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

suite = full_suite()

train_ds = Dataset(X_train.values, label=np.ravel(y_train), cat_features=[])
test_ds = Dataset(X_test.values, label=y_test, cat_features=[])


# evaluation_suite = model_evaluation()
# suite_result = evaluation_suite.run(train_ds, test_ds, linear_svc1000)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
# suite_result.show()

suite.run(train_ds, test_ds, linear_svc1000)


# In[31]:


get_ipython().system('pip install deepchecks')


# In[27]:


pred_rfc = linear_svc1000.predict(X_test.values)
accuracy = accuracy_score(y_test, pred_rfc)
print(accuracy)


# In[28]:


y_train_pred = linear_svc1000.decision_function(X_train.values)    
y_test_pred = linear_svc1000.decision_function(X_test.values) 


# In[29]:


train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.subplots(1, figsize=(10,7))
plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))

plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.savefig('SVM-ROC Curve 1.png', bbox_inches='tight')
plt.show()


# In[32]:


pip install scikit-plot


# In[33]:


import scikitplot as skplt
y_probas = linear_svc1000.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_probas, plot_micro=False, plot_macro=False, figsize=(10,7))
plt.savefig('SVM-ROC Curve 2.png', bbox_inches='tight')
plt.show()


# In[34]:


get_ipython().system('pip install scikit-plot')


# In[35]:


from sklearn.metrics import precision_recall_curve

y_probas = linear_svc1000.predict_proba(X_test)[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, y_probas)

#create precision recall curve
# fig, ax = plt.subplots()
plt.subplots(1, figsize=(10,7))
plt.plot(recall, precision, color='purple',label="SVM")

#add axis labels to plot
plt.legend()
plt.title('Precision-Recall Curve')
plt.ylabel('Precision')
plt.xlabel('Recall')

#display plot
plt.savefig('SVM-Prec-Rec 1.png', bbox_inches='tight')
plt.show()


# In[36]:


import scikitplot as skplt
y_probas = linear_svc1000.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(y_test, y_probas, figsize=(10,7))
plt.savefig('SVM-Prec-Rec 2.png', bbox_inches='tight')
plt.show()


# In[37]:


print('Training set score: {:.4f}'.format(linear_svc1000.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(linear_svc1000.score(X_test, y_test)))


# In[38]:


classi_report = classification_report(y_test, pred_rfc)
print(classi_report)


# In[39]:


#cofusion matrix
plt.subplots(figsize=(10,7))

cf_matrix = confusion_matrix(y_test, pred_rfc)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot = True, annot_kws = {'size':15}, cmap = 'Pastel1')
plt.savefig('SVM-CFM.png', bbox_inches='tight')
plt.show()


# In[40]:


import pickle
# save
with open('model-svm.pkl','wb') as f:
    pickle.dump(linear_svc1000,f)
    
# joblib.dump(linear_svc1000, 'joblib-model-svm.pkl')


# In[ ]:




