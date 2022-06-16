import logging
import os
import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
import yaml

from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = "data\\processed"
PARAMS = yaml.safe_load(open(f'{os.path.join(PROJECT_DIR)}\\params.yaml'))["train"]
X_TRAIN = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\x_train.txt"
Y_TRAIN = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\y_train.txt"
CLF_OUT = 'models'


def get_data() -> Tuple[np.array, np.array]:
    """
    get data for training
    """
    return np.loadtxt(X_TRAIN, delimiter=','), np.loadtxt(Y_TRAIN, delimiter=',')


def train_clf(x_train, y_train) -> Tuple['sklearn.ensemble._forest.RandomForestClassifier']:
    """
    train rfl clf
    """
    clf = RandomForestClassifier(random_state=PARAMS['seed'], n_jobs=-1, \
        n_estimators=PARAMS['n_estimators'], min_samples_split=PARAMS['min_split'])
    return clf.fit(x_train, y_train)


def write_model(clf) -> None:
    """
    write model to file
    """
    logging.info("writing model to file")
    with open(f"{os.path.join(PROJECT_DIR, CLF_OUT)}\\rf_clf.pkl", "wb") as f_out:
        pickle.dump(clf, f_out)


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    logging.info("loading data")
    x_train, y_train = get_data()
    logging.info("training model")
    trained_clf = train_clf(x_train, y_train)
    write_model(trained_clf)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()