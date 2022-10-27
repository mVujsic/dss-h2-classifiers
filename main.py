import os
import argparse
from dotenv import load_dotenv
from processing import dataset_process as processing
from classifiers.logistic_regression import logistic_regression
from classifiers.decision_tree import decision_tree
from classifiers.random_forest import random_forest
from classifiers.utils import generate_report

# Supported algorithms
ALGORITHMS = [
              'logistic_regression',
              'decision_tree',
              'random_forest',
              'naive_bayes',
              'svm',
              'knn'
]


# Main function to execute chosen classifier
def main():
    data_dir_path = os.path.join(os.getcwd(), 'data')
    data_file_path = os.path.join(data_dir_path, os.environ.get("FILE_NAME"))

    if os.path.exists(data_file_path):
        print(f"Executing {args.algorithm} on data {data_file_path}")
        X_train, X_test, y_train, y_test = processing.parse_data(data_file_path)

        if args.algorithm == "logistic_regression":
            y_predicted, y_real = logistic_regression(X_train, X_test, y_train, y_test)
            generate_report(y_real, y_predicted, method=args.algorithm)
        elif args.algorithm == "decision_tree":
            y_predicted, y_real = decision_tree(X_train, X_test, y_train, y_test)
            generate_report(y_predicted, y_real, method=args.algorithm)
        elif args.algorithm == "random_forest":
            y_predicted, y_real = random_forest(X_train, X_test, y_train, y_test)
            generate_report(y_predicted, y_real, method=args.algorithm)
        elif args.algorithm == "naive_bayes":
            pass
        elif args.algorithm == "svm":
            pass
        elif args.algorithm == "knn":
            pass
    else:
        raise FileNotFoundError


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Homework2 classification algorithms.')

    parser.add_argument('algorithm',
                        help='algorithm to be executed.')
    args = parser.parse_args()
    if args.algorithm in ALGORITHMS:
        main()
    else:
        print(f"Chosen algorithm not supported ... supported classifiers are {','.join(ALGORITHMS)}.")

