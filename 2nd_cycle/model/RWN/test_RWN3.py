from RWN3 import RWN
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta
import time

def data_load(data_dir, test_dir):
    df_train = pd.read_csv(data_dir)
    df_test = pd.read_csv(test_dir)
    
    label = df_train.iloc[:,-1].values
    x_data = df_train.iloc[:,:-1].values
    test_y = df_test.iloc[:,-1].values
    test_x = df_test.iloc[:,:-1].values
    train_x,val_x,train_y,val_y = train_test_split(x_data,label,test_size=0.2)

    return [train_x,train_y,val_x,val_y,test_x,test_y]

def main():
    dataset = data_load("total_sampled_data.csv","data_test.csv")
    x_train,y_train = dataset[:2]
    x_val,y_val = dataset[2:4]
    x_test,y_test = dataset[4:]
    
    model = RWN(x_train.shape[1]
                , 4 # number of stages
                , 256 # Dense output size in dense_block
                , 2 # nb_class
                , "ws" # graph_model
                , [32, 4, 0.75] # graph_parameter
               ).build_model()
    
    optimizer = Adam(learning_rate = 0.001)
    model.compile(optimizer = optimizer
        , loss = 'sparse_categorical_crossentropy'
        , metrics = ["accuracy"])

    model.fit(x_train, y_train, validation_data = (x_val, y_val), batch_size = 50, epochs = 100)

if __name__ == "__main__":
    main()

