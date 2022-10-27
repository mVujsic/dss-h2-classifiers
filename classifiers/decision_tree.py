from sklearn.tree import DecisionTreeClassifier


def decision_tree(X_train, X_test, y_train, y_test):
    print("Decision tree: min_samples_leaf=11")
    dtc = DecisionTreeClassifier(min_samples_leaf=11, class_weight={0:0.8, 1: 1.0})
    dtc.fit(X_train, y_train)

    y_predicted_results_y = dtc.predict(X_test)
    return y_predicted_results_y, y_test
