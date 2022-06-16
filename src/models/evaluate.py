import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Tuple, Text
import yaml
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)
from matplotlib import pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = "data\\processed"
X_TEST = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\x_test.txt"
Y_TEST = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\y_test.txt"
CLF_OUT = "models/rf_clf.pkl"


def get_data() -> Tuple[np.array, np.array]:
    """
    get data for training
    """
    return np.loadtxt(X_TEST, delimiter=","), np.loadtxt(Y_TEST, delimiter=",")


def get_model() -> Tuple["sklearn.ensemble._forest.RandomForestClassifier"]:
    """ """
    with (open(os.path.join(PROJECT_DIR, CLF_OUT), "rb")) as f:
        model = pickle.load(f)
        return model


def get_predictions(model, x_test) -> np.array:
    """ """
    return model.predict(x_test)


def get_model_metrics(model, x_test, y_test) -> Tuple[dict, np.array]:
    """ """

    predictions = get_predictions(model, x_test)
    roc = roc_auc_score(y_test, predictions)
    cr = classification_report(predictions, y_test, output_dict=True)
    cm = confusion_matrix(predictions, y_test)
    tn,fp,fn,tp = cm.ravel()

    f1 = f1_score(y_true=y_test, y_pred=predictions, average='macro')
    print(f1)

    report = {"accuracy": cr["accuracy"],
    "f1": cr["macro avg"]["f1-score"],
    "class_0_precison": cr["0.0"]["precision"],
    "class_0_recall": cr["0.0"]["recall"],
    # "class_0_f1": cr["0.0"]["f1-score"],
    "class_1_precison": cr["1.0"]["precision"],
    "class_1_recall": cr["1.0"]["recall"],
    # "class_1_f1": cr["1.0"]["f1-score"],
    "roc": roc,
    # "tp": int(tp),
    # "fp":int(fp),
    # "fn": int(fn),
    # "tn":int(tn),
    "cm":cm.tolist()}

    return report, cm


def evaluate_model(config_path: Text):
    """
    evaluate model on training data
    """
    logger = logging.getLogger(__name__)
    logger.info("getting model metrics")

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    x_test, y_test = get_data()
    clf = get_model()
    report, cm = get_model_metrics(clf, x_test, y_test)
    reports_folder = Path(config["evaluate"]["reports_dir"])
    metrics_path = reports_folder / config["evaluate"]["metrics_file"]

    json.dump(obj=report, fp=open(metrics_path, "w"))

    confusion_matrix_png_path = (
        reports_folder / config["evaluate"]["confusion_matrix_image"]
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="Blues")
    plt.savefig(os.path.join(PROJECT_DIR, confusion_matrix_png_path))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
