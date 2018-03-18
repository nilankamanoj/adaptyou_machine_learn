import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import cPickle


df = pd.read_csv('user.csv')

def label_df(df):
    avgs = []
    for column in list(df)[1:]:
        avgs.append(int(sum(df[column].tolist())/len(df[column].tolist())))
    labels=[]
    for j in range(len(df['user name'].tolist())):
        label=''
        for k in range(len(list(df)[1:])):
            if(int(df[list(df)[k+1]].tolist()[j])>= avgs[k]):
                label += str(k+1)+','
        if(len(label)==0):
            label='default,'
        labels.append(label[:-1])

    df['labels'] = pd.Series(labels)
    return df


def get_trains_tests(df,ratio):
    X, y = df.iloc[:,1:-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=1)
    return X_train, X_test, y_train, y_test

def test(model,X_test,Y_test,df):
    predict = model.predict(X_test).tolist()
    actual = Y_test.tolist()
    comp_count = len(list(df)[1:-1])
    total = [0] * (comp_count+1)
    success = [0] * (comp_count+1)
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

def save_obj(obj,name):
    with open(r"trained/"+name+".pickle", "wb") as output_file:
        cPickle.dump(obj, output_file)
        
def load_obj(name):
    with open(r"trained/"+name+".pickle", "rb") as input_file:
        return cPickle.load(input_file)   

df = label_df(df)
X_train, X_test, y_train, y_test = get_trains_tests(df,0.2)

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

print(test(nb,X_test,y_test,df))
print(test(dt,X_test,y_test,df))
print(test(lr,X_test,y_test,df))
print(test(rf,X_test,y_test,df))
print(test(ab,X_test,y_test,df))




