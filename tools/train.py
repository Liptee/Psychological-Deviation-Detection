from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tools.utils.saver import make_dir_if_not_exist 
from tools.utils.autoencoders import Autoencoder_ver2, Autoencoder_ver1
from tools.utils.saver import load_data
from tools.realtime_detect import anomaly_rowtime

import pickle
import json
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

def train(data_file: str, pipelines:dict, test_size: float = 0.5, random_state: int = 42, test: bool = False):
    file_name = data_file.split('.')[0].split('/')[-1]
    df = pd.read_csv(data_file)
    df = df.fillna(0)
    X = df.drop("class", axis=1)
    y = df["class"]
    del df

    make_dir_if_not_exist("models")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state, shuffle=True)
    
    for algo, pipeline in pipelines.items():
        model_file_name = f"{file_name}_{algo}.pkl"
        model = pipeline.fit(x_train, y_train)
        if test:
            pred = model.predict(x_test)
            print(f"for {algo} algorythm:")
            print(f"accuracy score: {classification_report(y_test, pred)}")

        with open(f"models/{model_file_name}", "wb") as f:
            pickle.dump(model, f)
# -------------------------------------------------------------------------------------------------------------------------

def train_timelaps(data_file: str, pipelines: dict, num_neighboor_frames: list = [-3, -1], test_size: float = 0.1, random_state: int = 42, test: bool = False):
    file_name = data_file.split('.')[0].split('/')[-1]
    df = pd.read_csv(data_file)
    df = df.fillna(0)
    df_copy = df.copy()
    #create timelaps data from df
    for i in reversed(num_neighboor_frames):
        for col in df.columns:
            df_copy[f"{col}_{i}"] = df[col].shift(i)
    #clear rows with NaN
    df_copy = df_copy.dropna()
    y = df_copy["class"]
    
    X = df_copy.drop("class", axis=1)
    del df
    del df_copy
    for column in X.columns:
        if column.startswith("class"):
            X.drop(column, axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state, shuffle=True)
    
    for algo, pipeline in pipelines.items():
        model_file_name = f"{file_name}_{algo}.pkl"
        model = pipeline.fit(x_train, y_train)
        if test:
            pred = model.predict(x_test)
            print(f"for {algo} algorythm:")
            print(f"accuracy score: {classification_report(y_test, pred)}")

        with open(f"models/timelaps_{model_file_name}", "wb") as f:
            pickle.dump(model, f)
# -------------------------------------------------------------------------------------------------------------------------

def autoencoder(data_file: str,
                epochs: int,
                model_name: str = None,
                train_size: float = 0.9,
                autoencoder=Autoencoder_ver1,
                hidden_size:int = 64,
                learning_rate=1e-3,
                batch_size: int = 32,
                output_file: str = "model.pkl"):
    print(f"data_file: {data_file}")
    print(f"model_name: {model_name}")
    print(f"output_file: {output_file}")
    df = pd.read_csv(data_file)
    df = df.fillna(0)
    X = df.drop("class", axis=1)
    y = df["class"]
    del df

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    input_size = len(x_train[0])

    if not model_name:
        model = autoencoder(input_size=input_size)#, hidden_size=hidden_size)
    else:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        for batch in DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(x_train).float()), batch_size=batch_size):
            x_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        for batch in DataLoader(TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(x_test).float()), batch_size=batch_size):
            x_batch, y_batch = batch
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * x_batch.size(0)
            
        train_loss = train_loss / len(x_train)
        val_loss = val_loss / len(x_test)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    print(output_file)
    with open(output_file, "wb") as f:
        pickle.dump(model, f)
# -------------------------------------------------------------------------------------------------------------------------

def autoencode_timelaps(data_file: str,
                        output_file: str,
                        epochs: int,
                        model_name: str = None,
                        train_size: float = 0.9, 
                        autoencoder=Autoencoder_ver1, 
                        hidden_size:int = 64, 
                        learning_rate=1e-3, 
                        batch_size: int = 32, 
                        num_neighboor_frames: list = [-3, -1]):
    
    file_name = data_file.split('.')[0].split('/')[-1]

    num_neighboor_frames = sorted(num_neighboor_frames)
    df = pd.read_csv(data_file)
    df = df.fillna(0)
    df_copy = df.copy()
    for i in reversed(num_neighboor_frames):
        for col in df.columns:
            df_copy[f"{col}_{i}"] = df[col].shift(i)
    df_copy = df_copy.dropna()
    y = df_copy["class"]
    
    X = df_copy.drop("class", axis=1)
    del df
    del df_copy
    for column in X.columns:
        if column.startswith("class"):
            X.drop(column, axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    print(len(x_train[0]))
    input_size = len(x_train[0])

    if not model_name:
        model = autoencoder(input_size=input_size)#, hidden_size=hidden_size)
    else:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        for batch in DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(x_train).float()), batch_size=batch_size):
            x_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        for batch in DataLoader(TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(x_test).float()), batch_size=batch_size):
            x_batch, y_batch = batch
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * x_batch.size(0)
            
        train_loss = train_loss / len(x_train)
        val_loss = val_loss / len(x_test)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    with open(output_file, "wb") as f:
        pickle.dump(model, f)

def create_json_autoencoders(model_file,
                      dir: str,
                      output_data,
                      pose_landmarks=False,
                      pose_cut=False,
                      face_landmarks=False,
                      right_hand=False,
                      left_hand=False,
                      num_neighboor=None):
        X = load_data(dir, "mp4")
        all_data = {}
        for x in X:
            data = anomaly_rowtime(model_file,
                                   source=x,
                                   pose_landmarks=pose_landmarks,
                                   face_landmarks=face_landmarks,
                                   left_hand_landmarks=left_hand,
                                   right_hand_landmarks=right_hand,
                                   pose_cut=pose_cut,
                                   num_neighboor_frames=num_neighboor,
                                   show=False,
                                   return_data=True)

            all_data[x] = data

        json_data = json.dumps(all_data, indent=len(X))
        with open(output_data, 'w') as f:
            f.write(json_data)