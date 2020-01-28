from __future__ import absolute_import, division, print_function, unicode_literals
from misc import *

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler

"""
/TODO
    * Change familysize to a binary alone or not
    * Create a one hot encoding for every categorical column
"""
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
TRAIN = True
TEST = False

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


"""
    Drops columns that serves no purpose during feature extraction
"""
def drop_columns(df):
    # 'PassengerId'
    # , 'SibSp', 'Parch'
    column_drops = ['Name', 'Ticket']
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
    
    #df = normalize_age(df)
    
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

    #df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    #df['FamilySize'] = df['FamilySize'].map(FAMILY_MAPPING)

    return df
def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y

def kaggle_tensorflow(train_data, X_test):
    train_data = add_columns(train_data)
    X_test = add_columns(X_test)
    
    train_data = set_alone(train_data)
    X_test = set_alone(X_test)

    # TODO
    # Try normalizing age
    
    dummy_columns = ["Pclass", "Age"]
    train_data = dummy_data(train_data, dummy_columns)
    X_test  = dummy_data(X_test,  dummy_columns)

    X = drop_columns(train_data)
    X_test = drop_columns(X_test)
    
    id_test = test['PassengerId'].values
    id_test = id_test.reshape(-1, 1)


    y = X['Survived'].values
    y = y.astype(float).reshape(-1, 1)
    X = X.drop('Survived', axis=1)
    
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=0)

    print("\n", X.shape, y.shape, X_test.shape)
    print(list(X.columns.values))
    print(list(X_test.columns.values))
        
    seed = 7                        # for reproducible purpose
    input_size = X_train.shape[1]   # number of features
    learning_rate = 0.001           # most common value for Adam
    epochs = 8500

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        np.random.seed(seed)

        X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
        
        W1 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W1')
        b1 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b1')
        sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                    logits=sigm, name='loss'))
        train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred') # 1 if >= 0.5
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')
        
        init_var = tf.global_variables_initializer()

    train_feed_dict = {X_input: X_train, y_input: y_train}
    dev_feed_dict = {X_input: X_dev, y_input: y_dev}
    test_feed_dict = {X_input: X_test} # no y_input since the goal is to predict it

    if(TRAIN):
        sess = tf.Session(graph=graph)
        sess.run(init_var)
        cur_loss = sess.run(loss, feed_dict=train_feed_dict)
        train_acc = sess.run(acc, feed_dict=train_feed_dict)
        test_acc = sess.run(acc, feed_dict=dev_feed_dict)
        print('step 0: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                            cur_loss, 100*train_acc, 100*test_acc))
                            
        for step in range(1, epochs+1):
            sess.run(train_steps, feed_dict=train_feed_dict)
            cur_loss = sess.run(loss, feed_dict=train_feed_dict)
            train_acc = sess.run(acc, feed_dict=train_feed_dict)
            test_acc = sess.run(acc, feed_dict=dev_feed_dict)
            if step%100 != 0: # print result every 100 steps
                continue
        
            print('step {3}: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       cur_loss, 100*train_acc, 100*test_acc, step))

        y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)
        prediction = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1),
                          columns=['PassengerId', 'Survived'])
        
        prediction.to_csv("lr-tf-submission.csv",index=False)
       

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/Titanic_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/Titanic_Kaggle/Data/test.csv")

    # https://www.kaggle.com/abevallerian/titanic-with-tensorflow
    kaggle_tensorflow(train, test)

    