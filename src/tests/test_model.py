import os
import pickle
from pathlib import Path
import unittest
import sys
import numpy as np
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = "data\\processed"
PARAMS = yaml.safe_load(open(f"{os.path.join(PROJECT_DIR)}\\params.yaml"))["train"]
X_TRAIN = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\x_train.txt"
Y_TRAIN = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\y_train.txt"
X_TEST = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\x_test.txt"
Y_TEST = f"{os.path.join(PROJECT_DIR, PROCESSED_DIR)}\\y_test.txt"
CLF_MODEL = "models/rf_clf.pkl"


class SimplePipeline:
    """
    simple pipeline to
    train base model for
    comparison
    """

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.Y_test = None, None, None, None
        self.model = None
        self.load_dataset()

    def load_dataset(self):
        """
        load the dataset
        """
        self.X_train, self.X_test, = np.loadtxt(
            X_TRAIN, delimiter=","
        ), np.loadtxt(X_TEST, delimiter=",")
        self.y_train, self.y_test = np.loadtxt(Y_TRAIN, delimiter=","), np.loadtxt(
            Y_TEST, delimiter=","
        )

    def train(self, algorithm=DecisionTreeClassifier(random_state=42)):
        """
        train base model
        """
        self.model = algorithm
        self.model.fit(self.X_train, self.y_train)

    def predict(self, input_data):
        """
        predictions for base model
        """
        return self.model.predict(input_data)

    def get_accuracy(self):
        """
        accuracy score
        for base model
        """
        return self.model.score(X=self.X_test, y=self.y_test)

    def run_pipeline(self):
        """
        Helper method to run
        multiple pipeline methods
         with one call
        """

        self.load_dataset()
        self.train()


class NewModelPipeline:
    """
    used for newly trained
    model
    """

    def __init__(self):
        self.X_test, self.Y_test = None, None
        self.model = None
        self.load_model()
        self.load_dataset()

    def load_model(self):
        with (open(os.path.join(PROJECT_DIR, CLF_MODEL), "rb")) as f:
            self.model = pickle.load(f)

    def load_dataset(self):
        """
        load the dataset
        """
        self.X_test = np.loadtxt(X_TEST, delimiter=",")
        self.y_test = np.loadtxt(Y_TEST, delimiter=",")

    def predict(self, input_data):
        """
        get predictions
        """
        return self.model.predict(input_data)

    def get_accuracy(self):
        """
        get accuracy
        """
        return self.model.score(X=self.X_test, y=self.y_test)

    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_model()
        self.load_dataset()


class TestModelQuality(unittest.TestCase):
    def setUp(self):
        # We prepare both pipelines for use in the tests
        self.pipeline_v1 = SimplePipeline()
        self.pipeline_v1.run_pipeline()
        self.pipeline_v2 = NewModelPipeline()
        self.pipeline_v2.run_pipeline()

    def test_accuracy_higher_than_benchmark(self):
        """
        compare accuracy to
        benchmark
        """
        # given
        benchmark_accuracy = 0.65
        actual_accuracy = self.pipeline_v1.get_accuracy()

        # Then
        print(
            f"model accuracy: {round(actual_accuracy, 4)}, benchmark accuracy: {benchmark_accuracy}"
        )
        self.assertTrue(round(actual_accuracy, 4) > benchmark_accuracy)

    def test_f1_score_higher_than_benchmark(self):
        """
        >0.9 very good
        0.8 - 0.9 good
        0.5 - 0.8 ok
        < 0.5 not good
        """

        # given
        benchmark_f1_score = 0.5
        predictions = self.pipeline_v2.predict(self.pipeline_v1.X_test)

        actual_f1_score = f1_score(self.pipeline_v2.y_test, predictions)
        # Then
        print(
            f"model f1: {round(actual_f1_score, 4)}, benchmark f1: {benchmark_f1_score}"
        )
        self.assertTrue(round(actual_f1_score, 4) > benchmark_f1_score)

    def test_tpr_higher_than_benchmark(self):
        """
        compare tpr to benchmark
        """
        # given
        benchmark_tpr_score = 0.3
        predictions = self.pipeline_v2.predict(self.pipeline_v2.X_test)
        matrix_results = confusion_matrix(predictions, self.pipeline_v2.y_test)
        false_neg, true_pos = matrix_results.ravel()[2], matrix_results.ravel()[3]

        # sensitivity, hit rate, recall, or true positive rate
        actual_tpr = true_pos / (true_pos + false_neg)

        # then
        print(
            f"model tpr: {round(actual_tpr, 4)}, benchmark tpr: {benchmark_tpr_score}"
        )
        self.assertTrue(round(actual_tpr, 4) > benchmark_tpr_score)

    def test_tnr_higher_than_benchmark(self):
        """
        compare tnr to benchmark
        """
        # given
        benchmark_tnr_score = 0.3
        predictions = self.pipeline_v2.predict(self.pipeline_v2.X_test)
        matrix_results = confusion_matrix(predictions, self.pipeline_v2.y_test)
        true_neg, false_pos = matrix_results.ravel()[0], matrix_results.ravel()[1]

        # Specificity or true negative rate
        actual_tnr = true_neg / (true_neg + false_pos)

        # then
        print(
            f"model tnr: {round(actual_tnr, 4)}, benchmark tnr: {benchmark_tnr_score}"
        )
        self.assertTrue(round(actual_tnr, 4) > benchmark_tnr_score)

    def test_fnr_lower_than_benchmark(self):
        """
        compare fnr to benchmark
        """
        # given
        benchmark_fnr_score = 0.40
        predictions = self.pipeline_v2.predict(self.pipeline_v2.X_test)
        matrix_results = confusion_matrix(predictions, self.pipeline_v2.y_test)
        false_neg, true_pos = matrix_results.ravel()[2], matrix_results.ravel()[3]

        # False negative rate
        actual_fnr = false_neg / (true_pos + false_neg)

        # then
        print(f"model fnr: {round(actual_fnr,4)}, benchmark fnr: {benchmark_fnr_score}")
        self.assertTrue(round(actual_fnr, 4) < benchmark_fnr_score)

    def test_accuracy_compared_to_basemodel(self):
        """
        compare base model
        accuracy to newly trained
        """
        # given
        v1_accuracy = self.pipeline_v1.get_accuracy()
        v2_accuracy = self.pipeline_v2.get_accuracy()

        # Then
        # print(f'pipeline v1 accuracy: {v1_accuracy}')
        print(
            f"pipeline v2 accuracy: {round(v2_accuracy,4)} >= {round(v1_accuracy, 4)} pipeline v1 accuracy"
        )
        self.assertTrue(v2_accuracy >= v1_accuracy)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelQuality)
    unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
