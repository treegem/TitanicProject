from sklearn.ensemble import RandomForestClassifier

from src.utility.data_preparation import cleaned_split_data


def main():
    data_train, data_val, y_train, y_val = cleaned_split_data()
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data_train, y_train)
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    print('score_train:', score_train, '; score_val:', score_val)


if __name__ == "__main__":
    main()
