# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:13:22 2018

@author: Polestar User
"""

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

train = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\train.csv')
test1 = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\test.csv')

data = train.append(test)


plot = test.isna().sum()
plot.plot(kind = 'bar')

data['Fare'] = data['Fare'].fillna(5)

data['Embarked'] = data['Embarked'].fillna('S') 

data["Cabin"] = data["Cabin"].fillna(0)

data["Cabin"] = np.where(data["Cabin"]==0,0,1)



data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data['Title'], data['Sex'])

title_dict = {'Capt':0, 'Col':0,
    'Don':0, 'Dr':0, 'Major':0, 'Rev':0, 'Jonkheer':0, 'Dona':0,'Countess':1, 'Lady':1, 'Sir':1,'Mlle':2, 'Miss':2,'Ms':2, 'Miss':2,'Mme':3, 'Mrs':3,"Master":4,"Mr":5}

data['Title'].replace(title_dict,inplace=True)


#data['abc'] = data['Name'].apply(lambda l: 'Mrs' if ("Mrs" in l) else (("Mr" if ("Mr" in l) else ("Master" if ("Master" in l) else "Miss")))) 


a = data['Age'].mean()

data['Age'] = data['Age'].fillna(0)

#def f(x,y):
#    if x =='Mrs' and y==0:
#        return 36.804598
#    elif x =='Mr' and y ==0:
#        return 32.280928
#    elif x =='Miss' and y ==0:
#        return 24.361139
#    elif x =='Master' and y ==0:
#        return 5.482642
#    else:
#        return y



data['Age'] = data.apply(lambda l: f(l['abc'],l['Age']),axis=1)

del data['abc']

del data['PassengerId']

del data['Name']

del data['Ticket']


data.columns.values

train1 = data[0:890]
test1 = data[891:]

del test1['Survived']

X = train1.copy()

y = train1['Survived']

del X['Survived']

train1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\train1.csv')

test1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\test1.csv')

X= pd.get_dummies(X)

test1=pd.get_dummies(test1)


clf = LogisticRegressionCV(cv=10, random_state=0,penalty='l1',solver = 'liblinear').fit(X, y)

clf.score(X,y)

predtest = clf.predict(test1)

sub1 = pd.DataFrame()
sub1['PassengerId'] = test['PassengerId']
sub1['Survived'] = predtest
sub1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\LogisticSub2.csv',index=False)



##############################################################################

# Data after imputing missing values

#############################################################################


train = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\train1.csv')
test = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\test1.csv')

data = train.append(test)

r = [0,16,32,48,64,120]
g = [0,1,2,3,4]
data['Age'] = pd.cut(data['Age'],bins = r,labels=g)


p = [0,7.91,14.454,31,520]
q = [0,1,2,3]
data['NewFare'] = pd.cut(data['Fare'],bins = p,labels=q)

data['NewFare']=data['NewFare'].fillna(1)

del data['Fare']

data['Sex'] = np.where(data['Sex']=="male",1,0)

emd_dict = {"S":1,"C":2,"Q":3}
data['Embarked'].replace(emd_dict,inplace = True)

train1 = data[0:890]
test1 = data[891:]

train1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\train1.csv')

test1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\test1.csv')

data.isna().sum()


X = train1.copy()

y = train1['Survived']

del X['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0,test_size=0.30)


clf = LogisticRegression(random_state=0).fit(X_train,y_train)


###########################################################################
            # Tree
############################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import tree

train = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\train1.csv')
test = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\test1.csv')



X = train.copy()

y = train['Survived']

del X['Survived']

del test['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0,test_size=0.30)


model = tree.DecisionTreeClassifier(criterion='gini').fit(X_train,y_train)

model.score(X_test, y_test)

pred = model.predict(X_test)

cm = confusion_matrix(y_test,pred)



sub1 = pd.DataFrame()
sub1['PassengerId'] = test1['PassengerId']
sub1['Survived'] = pred
sub1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\LogisticSub3.csv',index=False)



parameters = {'max_depth':range(3,20)}
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
clf.fit(X, y)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 
pred=clf.predict(test)
sub1 = pd.DataFrame()
sub1['PassengerId'] = test1['PassengerId']
sub1['Survived'] = pred
sub1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\LogisticSub4.csv',index=False)

##########################################################################

##########################################################################

train = pd.read_csv('C:\\Users\\Polestar User\\Desktop\\train.csv')
test = pd.read_csv('C:\\Users\\Polestar User\\Desktop\\test.csv')

data = train.append(test)


train1 = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\train.csv')
test1 = pd.read_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\test.csv')

data1 = train1.append(test1)

data1.isna().sum()
 
#data1['Alone'] = np.where((data1['SibSp']==0) and (data1['Parch']==0),1,0)


def f(x,y):
    if x ==0 and y==0:
        return 1
    else:
        return 0


data1['Alone'] = data1.apply(lambda l: f(l['SibSp'],l['Parch']),axis=1)

data['Alone'] = data1['Alone'].copy()

train00 = data[0:890]
test01 = data[891:]

from sklearn.ensemble import RandomForestClassifier


X = train.copy()

y = train['Survived']

del X['Survived']

del test['Survived']

model = tree.DecisionTreeClassifier(criterion='gini').fit(X,y)


model = RandomForestClassifier(n_estimators = 1000, random_state=0).fit(X,y)

pred = model.predict(test)

sub1 = pd.DataFrame()
sub1['PassengerId'] = test1['PassengerId']
sub1['Survived'] = pred
sub1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\RandomSub1.csv',index=False)



parameters = {'max_depth':range(3,20)}
clf = GridSearchCV(RandomForestClassifier(n_estimators=100,min_samples_split=15,min_samples_leaf=8,max_features=5), parameters, n_jobs=10,cv = 2)
clf.fit(X, y)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 

pred = clf.predict(test)

sub1 = pd.DataFrame()
sub1['PassengerId'] = test1['PassengerId']
sub1['Survived'] = pred
sub1.to_csv('C:\\Users\\Polestar User\\Documents\\Classification\\Titanic\\RandomForestSub3.csv',index=False)



