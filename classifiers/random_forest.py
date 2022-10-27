from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, X_test, y_train, y_test):
    print("Random forest: min_samples_leaf=11")
    rf = RandomForestClassifier(min_samples_leaf=15,
                                class_weight={0:0.6, 1: 1.0},
                                n_estimators=150)
    rf.fit(X_train, y_train)

    predicted_results_y = rf.predict(X_test)
    return predicted_results_y, y_test

