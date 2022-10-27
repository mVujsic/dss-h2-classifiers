import random
import pandas
from sklearn.model_selection import train_test_split


def parse_data(data_file_path):
    df = pandas.read_csv(data_file_path)
    df = df.reset_index(drop=True)
    dataset = _normalize_data(df)
    return _split_test_train_set(dataset)


def _normalize_data(df):
    df['Gender'] = df.apply(lambda row: 0 if row['Gender'] == 'Female' else 1, axis=1)
    df = df.drop(columns=['User ID'], axis=1)
    return df


def _split_test_train_set(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Purchased', axis=1),
                                                        df['Purchased'],
                                                        test_size=0.2,
                                                        random_state=random.randint(10, 332))
    return X_train, X_test, y_train, y_test

