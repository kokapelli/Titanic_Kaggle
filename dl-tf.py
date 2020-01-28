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

    
def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

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

def build_neural_network(classes, hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, classes])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    fc = tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def kaggle_tensorflow(train_data, test_data):
    train_data = add_columns(train_data)
    test_data = add_columns(test_data)
    
    test_passenger_id = test_data["PassengerId"]
    
    train_data = set_alone(train_data)
    test_data = set_alone(test_data)

    # TODO
    # Try normalizing age
    
    dummy_columns = ["Pclass"]
    train_data = dummy_data(train_data, dummy_columns)
    test_data  = dummy_data(test_data,  dummy_columns)

    train_data = drop_columns(train_data)
    test_data = drop_columns(test_data)    
    train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)

    model = build_neural_network(train_x.shape[1])

    print(train_data.head())
        
    epochs = 400
    train_collect = 50
    train_print=train_collect*2

    learning_rate_value = 0.0001
    batch_size=32

    x_collect = []
    train_loss_collect = []
    train_acc_collect = []
    valid_loss_collect = []
    valid_acc_collect = []

    if(TRAIN):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration=0
            for e in range(epochs):
                for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
                    iteration+=1
                    feed = {model.inputs: train_x,
                            model.labels: train_y,
                            model.learning_rate: learning_rate_value,
                            model.is_training:True
                        }

                    train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
                    
                    if iteration % train_collect == 0:
                        x_collect.append(e)
                        train_loss_collect.append(train_loss)
                        train_acc_collect.append(train_acc)

                        if iteration % train_print==0:
                            print("Epoch: {}/{}".format(e + 1, epochs),
                            "Train Loss: {:.4f}".format(train_loss),
                            "Train Acc: {:.4f}".format(train_acc))
                                
                        feed = {model.inputs: valid_x,
                                model.labels: valid_y,
                                model.is_training:False
                            }
                        val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                        valid_loss_collect.append(val_loss)
                        valid_acc_collect.append(val_acc)
                        
                        if iteration % train_print==0:
                            print("Epoch: {}/{}".format(e + 1, epochs),
                            "Validation Loss: {:.4f}".format(val_loss),
                            "Validation Acc: {:.4f}".format(val_acc))
                        

            saver.save(sess, "./titanic.ckpt")

        plt.plot(x_collect, train_loss_collect, "r--")
        plt.plot(x_collect, valid_loss_collect, "g^")
        plt.show()

        plt.plot(x_collect, train_acc_collect, "r--")
        plt.plot(x_collect, valid_acc_collect, "g^")
        plt.show()

        if(TEST):
            model=build_neural_network(train_x.shape[1])
            restorer=tf.train.Saver()
            with tf.Session() as sess:
                restorer.restore(sess,"./titanic.ckpt")
                feed={
                    model.inputs:test_data,
                    model.is_training:False
                }
                test_predict=sess.run(model.predicted,feed_dict=feed)

            test_predict[:10]
            
            binarizer=Binarizer(0.5)
            test_predict_result=binarizer.fit_transform(test_predict)
            test_predict_result=test_predict_result.astype(np.int32)
            test_predict_result[:10]

            passenger_id=test_passenger_id.copy()
            evaluation=passenger_id.to_frame()
            evaluation["Survived"]=test_predict_result
            evaluation[:10]

            evaluation.to_csv("tf_submission.csv",index=False)

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/Titanic_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/Titanic_Kaggle/Data/test.csv")
    kaggle_tensorflow(train, test)

    