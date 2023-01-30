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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    
    print(x_train)
    for algo, pipeline in pipelines.items():
        model_file_name = f"{file_name}_{algo}.pkl"
        model = pipeline.fit(x_train, y_train)
        if test:
            pred = model.predict(x_test)
            print(f"for {algo} algorythm:")
            print(f"accuracy score: {classification_report(y_test, pred)}")

        with open(f"models/{model_file_name}", "wb") as f:
            pickle.dump(model, f)

