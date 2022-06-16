# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = "data\\raw\\indians_diabetes_data.csv"
DATA_URL = \
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def get_data() -> None:
    """
    get data for testing
    """

    names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "target"]
    data = pd.read_csv(DATA_URL, names=names)
    data.to_csv(os.path.join(PROJECT_DIR, RAW_DIR), index=False)


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    get_data()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
