from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

"""
/TODO
    * Change familysize to a binary alone or not
    * Create a one hot encoding for every categorical column
"""
#survival	Survival	        0 = No, 1 = Yes
#pclass	    Ticket class	    1 = 1st, 2 = 2nd, 3 = 3rd
#sex	    Sex	
#Age	    Age in years	
#sibsp	    # of siblings / spouses aboard the Titanic	
#parch	    # of parents / children aboard the Titanic	
#ticket	    Ticket number	
#fare	    Passenger fare	
#cabin	    Cabin number	
#embarked	Port of Embarkation	 C = Cherbourg, Q = Queenstown, S = Southampton

fc = tf.feature_column
CATEGORICAL_COLUMNS = ['Sex', 'SibSp', 'Parch', 'Pclass', 'Cabin',
                    'Embarked', 'Alone', 'Survived']
NUMERIC_COLUMNS = ['Age', 'Fare']
TITLE_MAPPING = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
SEX_MAPPING = {"male": 0, "female": 1}
EMBARKED_MAPPING = {"S": 0, "C": 1, "Q": 2}
CABIN_MAPPING = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
FAMILY_MAPPING = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
SUBMIT = True

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocab))

def set_alone(df):
    alone = list()
    for index, row in df.iterrows():
        if(row['SibSp'] == 0 and row['Parch'] == 0):
            alone.append(1)
        else:
            alone.append(0)

    df['Alone'] = alone

    return df

def display_plots(df):
    plt.figure('Age')
    df.Age.hist(bins=20)
    plt.figure('Sex')
    df.Sex.value_counts().plot(kind='barh')
    plt.figure('Alone')
    df.Alone.value_counts().plot(kind='barh')
    plt.figure('Class')
    df.Pclass.value_counts().plot(kind='barh')
    plt.figure('Embarked')
    df.Embarked.value_counts().plot(kind='barh')
    plt.figure('Survival')
    df.groupby('Sex').Survived.mean().plot(kind='barh')


    plt.show()

"""
    Drops columns that serves no purpose during feature extraction
"""
def drop_columns(df):
    # 'PassengerId'
    column_drops = ['Name', 'Ticket', 'SibSp', 'Parch']
    df = df.drop(column_drops, axis=1)

    return df

"""
    Adds additional columns for feature extraction
"""
def add_columns(df):

    # Check if person was alone or not
    #df['Alone'] = set_alone(df)
    # Extract title of persono
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].map(TITLE_MAPPING)
    df['Sex'] = df['Sex'].map(SEX_MAPPING)

    
    df["Age"].fillna(df.groupby("Title")["Age"].transform("median"), inplace=True)
    df.loc[ df['Age'] <= 16, 'Age'] = 0,
    df.loc[(df['Age'] > 16) & (df['Age'] <= 26), 'Age'] = 1,
    df.loc[(df['Age'] > 26) & (df['Age'] <= 36), 'Age'] = 2,
    df.loc[(df['Age'] > 36) & (df['Age'] <= 62), 'Age'] = 3,
    df.loc[ df['Age'] > 62, 'Age'] = 4
    
    #normalize_age(df)

    
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map(EMBARKED_MAPPING)

    df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    df.loc[ df['Fare'] <= 17, 'Fare'] = 0,
    df.loc[(df['Fare'] > 17) & (df['Fare'] <= 30), 'Fare'] = 1,
    df.loc[(df['Fare'] > 30) & (df['Fare'] <= 100), 'Fare'] = 2,
    df.loc[ df['Fare'] > 100, 'Fare'] = 3

    df['Cabin'] = df['Cabin'].str[:1]
    df['Cabin'] = df['Cabin'].map(CABIN_MAPPING)
    df["Cabin"].fillna(df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df['FamilySize'] = df['FamilySize'].map(FAMILY_MAPPING)

    return df

def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data

def get_column_list(data):
    column_list = list()
    for col in data.columns: 
        column_list.append(col)

    return column_list

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))

    return data

def kaggle_scikit(train, test):
    train = add_columns(train)
    test = add_columns(test)    
    
    train = drop_columns(train)
    test = drop_columns(test)
    train = train.drop('PassengerId', axis=1)
    
    train_data = train.drop('Survived', axis=1)
    target = train['Survived']
    #print(train_data.head(30))
    #print(train_data.shape, target.shape)

    print(train_data.head())
    print(test.head())

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = SVC()
    scoring = 'accuracy'
    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

    print(score)
    print(round(np.mean(score)*100,2))

    if(SUBMIT):
        clf = SVC()
        clf.fit(train_data, target)

        test_data = test.drop("PassengerId", axis=1).copy()
        prediction = clf.predict(test_data)

        submission = pd.DataFrame({
            "PassengerId": test["PassengerId"],
            "Survived": prediction
        })

        submission.to_csv('submission.csv', index=False)

        submission = pd.read_csv('submission.csv')
        submission.head()

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/Titanic_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/Titanic_Kaggle/Data/test.csv")
    
    # https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
    kaggle_scikit(train, test)