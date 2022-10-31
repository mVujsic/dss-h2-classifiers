import matplotlib.pyplot as plt


def visualized(x_1, x_2, y, x_label, y_label):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i, item in enumerate(y):
        if item == 0:
            x1.append(x_1[i])
            y1.append(x_2[i])
        else:
            x2.append(x_1[i])
            y2.append(x_2[i])

    a1 = plt.scatter(x1, y1, label='klasa1')
    a2 = plt.scatter(x2, y2, label='klasa2')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend((a1, a2), ('klasa1', 'klasa2'))
    plt.legend(bbox_to_anchor=(1.28, 0.9), loc='center right')
    plt.show()
