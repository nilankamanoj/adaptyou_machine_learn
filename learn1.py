import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier


df = pd.read_csv('user.csv')

def label_df(df):
    avgs = []
    for i in range(5):
        avgs.append(int(sum(df['comp'+str(i+1)].tolist())/len(df['comp'+str(i+1)].tolist())))
    labels=[]
    for j in range(1024):
        label=''
        for k in range(5):
            if(int(df['comp'+str(k+1)].tolist()[j])>= avgs[k]):
                label += str(k+1)+','
        if(len(label)==0):
            label='default,'
        labels.append(label[:-1])

    df['labels'] = pd.Series(labels)
    return df


def get_trains_tests(df):
    X, y = df.iloc[:,1:-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def test(model,X_test,Y_test):
    predict = model.predict(X_test).tolist()
    actual = Y_test.tolist()
    total = [0,0,0,0,0,0]
    success = [0,0,0,0,0,0]
    for i in range(len(predict)):
        if(actual[i]=='default'):
            total[0]+=1
            if(predict[i]=='default'):
                success[0]+=1
        elif(actual[i] == predict[i]):
            success[len(actual[i].split(','))]+=1
        if(actual[i]!='default'):
            total[len(actual[i].split(','))]+=1
    return total,success,sum(total),sum(success),float(sum(success))/float(sum(total))
    
    

df = label_df(df)
X_train, X_test, y_train, y_test = get_trains_tests(df)

nb=GaussianNB()
dt=DecisionTreeClassifier()
lr=LogisticRegression()
rf=RandomForestClassifier()
ab=AdaBoostClassifier()

nb.fit(X_train,y_train)
dt.fit(X_train,y_train)
lr.fit(X_train,y_train)
rf.fit(X_train,y_train)
ab.fit(X_train,y_train)


print(test(nb,X_test,y_test))
print(test(dt,X_test,y_test))
print(test(lr,X_test,y_test))
print(test(rf,X_test,y_test))
print(test(ab,X_test,y_test))




