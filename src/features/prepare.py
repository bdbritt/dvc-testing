import logging
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = "data\\raw\\indians_diabetes_data.csv"
PROCESSED_DIR = "data\\processed"
PARAMS = yaml.safe_load(open(f'{os.path.join(PROJECT_DIR)}\\params.yaml'))["prepare"]
OUT_X_TRAIN = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\x_train.txt"
OUT_X_TEST = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\x_test.txt"
OUT_Y_TRAIN = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\y_train.txt"
OUT_Y_TEST = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\y_test.txt"


def get_data() -> pd.DataFrame:
    """
    get data for testing
    """

    return pd.read_csv(os.path.join(PROJECT_DIR, RAW_DIR))


def split_data(data) -> None:
    """
    split data into
    training and validation
    """

    X = data.iloc[:, :-1].values
    y = data.loc[:, "target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PARAMS["split"], random_state=PARAMS["seed"]
    )

    logging.info("writing data to file")
    np.savetxt(os.path.join(PROJECT_DIR, OUT_X_TRAIN), X_train, delimiter=',')
    np.savetxt(os.path.join(PROJECT_DIR, OUT_X_TEST),  X_test, delimiter=',')
    np.savetxt(os.path.join(PROJECT_DIR, OUT_Y_TRAIN),  y_train, delimiter=',')
    np.savetxt(os.path.join(PROJECT_DIR, OUT_Y_TEST),  y_test, delimiter=',')


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    data_loaded = get_data()
    split_data(data_loaded)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
