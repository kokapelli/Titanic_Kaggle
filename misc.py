import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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