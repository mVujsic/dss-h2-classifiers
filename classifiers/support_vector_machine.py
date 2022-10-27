from sklearn.svm import SVC


def svm(X_train, X_test, y_train, y_test):
    print("Support vector machine:  ")

    svc = SVC(kernel='rbf', C=2)
    svc.fit(X_train, y_train)

    y_predict_result = svc.predict(X_test)

    return y_predict_result, y_test