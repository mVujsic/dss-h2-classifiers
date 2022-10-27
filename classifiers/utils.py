from sklearn.metrics import classification_report
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def generate_report(y_pred, y_real, method='logistic_regression'):
    out_class_labels = ["Предвиђена 0", 'Предвиђена 1']
    general_report = classification_report(y_real, y_pred, target_names=out_class_labels)
    print(general_report)

    classes = np.unique(y_real)

    fig, ax = plt.subplots()
    conf_matrix = metrics.confusion_matrix(y_real, y_pred, labels=classes)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Предвиђени", ylabel="Тачни", title=method)
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.show()
