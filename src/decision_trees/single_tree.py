from scipy.stats import uniform
from sklearn.tree import DecisionTreeClassifier

from src.utility.data_preparation import load_clean_data, split_data
from utility.data_preparation import standard_scale
from utility.parameter_search import randomized_search_cv


def main():
    data = load_clean_data()
    y = data.pop('Survived')
    data = standard_scale(data)
    clf = DecisionTreeClassifier()

    param_dist = {'max_depth': list(range(1, 50)) + [None],
                  'min_samples_split': list(range(2, 10)),
                  'min_samples_leaf': list(range(1, 10)),
                  'max_features': list(range(1, 10)) + [None],
                  'min_impurity_decrease': uniform(loc=0, scale=1)}
    clf = randomized_search_cv(clf, data, param_dist, y, n_jobs=4)  # n_jobs = number of cores used, adjust for dual

    data = load_clean_data()
    data_train, data_val, y_train, y_val = split_data(data)
    clf.fit(data_train, y_train)
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    print('score_train:', score_train, '; score_val:', score_val)


if __name__ == '__main__':
    main()
