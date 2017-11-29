from sklearn.tree import DecisionTreeClassifier

from src.utility.data_preparation import cleaned_split_data
from utility.data_preparation import standard_scale


def main():
    data_train, data_val, y_train, y_val = cleaned_split_data()
    data_train = standard_scale(data_train)
    data_val = standard_scale(data_val)
    clf = DecisionTreeClassifier()
    clf.fit(data_train, y_train)
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    print('score_train:', score_train, '; score_val:', score_val)


if __name__ == '__main__':
    main()
