from sklearn.neighbors import KNeighborsClassifier


def knn(X_train, X_test, y_train, y_test):
    print("KNN- n_neighbors=5, algorithm='kd_tree'")
    knn_model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', p=2) # algorithm='ball_tree')
    knn_model.fit(X_train, y_train)

    y_prediction_results = knn_model.predict(X_test)
    return y_prediction_results, y_test