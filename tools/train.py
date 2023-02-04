from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tools.utils.saver import make_dir_if_not_exist 

import pickle
import pandas as pd


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
    print(list(X.columns))

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