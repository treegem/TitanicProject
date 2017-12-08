from sklearn.neighbors import KNeighborsClassifier

from src.utility.data_preparation import load_clean_data, standard_scale, split_data


def main():
    data = load_clean_data()
    data_train, data_val, y_train, y_val = split_data(data)

    clf = KNeighborsClassifier()
    clf.fit(data_train, y_train)
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    print('\nscore_train:', score_train, '; score_val:', score_val)


if __name__ == '__main__':
    main()
