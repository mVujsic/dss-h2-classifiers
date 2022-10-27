from sklearn import linear_model


def logistic_regression(X_train, X_test, y_train, y_test):
    print("Logistic regression: solver=liblinear, max_iter=200, class_weight={0:0.8, 1:1}, C=0.7")
    model = linear_model.LogisticRegression(class_weight={0:0.8, 1:1}, C=0.7,
                                            solver='liblinear',
                                            max_iter=200)
    model.fit(X_train, y_train)
    predicted_results_y = model.predict(X_test)
    return predicted_results_y, y_test

