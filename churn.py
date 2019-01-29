# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# loading the train, check and test datasets
train = pd.read_csv('churn_train.csv',index_col=False)
test = pd.read_csv('churn_test.csv',index_col=False)

# knowing the column names
column_names = train.columns

# removing the unnecessary labels from the dataset
train.drop(labels = ['st',' acclen',' phnum'], axis = 1,inplace = True)
test.drop(labels = ['st',' acclen',' phnum'], axis = 1, inplace = True)


# enocding multiple categorical columns using LabelEncoder 
labels = [' intplan',' voice',' label']
from sklearn.preprocessing import LabelEncoder
def multi_label_encoding(df,cat_var):

    for index, cat_feature in enumerate(cat_var): 
        le = LabelEncoder()
        le.fit(df.loc[:, cat_feature])    
        df.loc[:, cat_feature] = le.transform(df.loc[:, cat_feature])

    return df

train = multi_label_encoding(train,labels)
test = multi_label_encoding(test, labels)


# splitting the data
y_train = train[' label']
X_train = train.drop(labels = ' label', axis = 1)
y_test = test[' label']
X_test = test.drop(labels = ' label',axis = 1)

# Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# Feature Selection using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Comparing Algorithms
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = []
models.append(( 'LR' , LogisticRegression()))
models.append(( 'KNN' , KNeighborsClassifier()))
models.append(( 'CART' , DecisionTreeClassifier()))
models.append(( 'NB' , GaussianNB()))
models.append(( 'SVM' , SVC()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# plotting the results   
plt.figure()
plt.boxplot(results)
plt.xlabel(names)
plt.show()

# SVC
clf = SVC(random_state=0)
clf.fit(X_train,y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accurcay is :', accuracy_score(y_test, y_pred))

# Algorithm Tuning
from sklearn.model_selection import GridSearchCV
c = [.1,1,10]
gammas = [.01,.1,1]
kernels = ['linear','rbf']
params = {'C':c,'kernel':kernels,'gamma':gammas}
grid = GridSearchCV(estimator=clf,param_grid=params,cv=3)
grid.fit(X_train,y_train)
print(grid.best_params_)


# final model 
clf = SVC(C=1.0,kernel='rbf',gamma=0.1,random_state=0)
clf.fit(X_train,y_train)
print('Final Model Accurcay is :', accuracy_score(y_test, clf.predict(X_test)))

    
# save the model to disk
from sklearn.externals.joblib import dump
filename = ' finalized_model.sav '
dump(clf, filename)
print('Model Saved to disk')









    

