from sklearn.naive_bayes import GaussianNB


def naive_bayes(X_train, X_test, y_train, y_test):
    print("Bayes - plain")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_prediction_results = gnb.predict(X_test)
    return y_prediction_results, y_test