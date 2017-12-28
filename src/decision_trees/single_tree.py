from scipy.stats import uniform
from sklearn.tree import DecisionTreeClassifier

from src.utility.data_preparation import load_clean_data, load_clean_split_standard_data
from src.utility.data_preparation import standard_scale
from src.utility.parameter_search import randomized_search_cv
from src.utility.storage_utility import save_model


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
    clf = randomized_search_cv(clf, data, parameter_distribution=param_dist,
                               targets=y, n_jobs=3, n_iter=1000, cv=5,
                               verbose=True)

    data_train, data_val, y_train, y_val = load_clean_split_standard_data()
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    save_model(clf, file_name='single_tree')

    print('\nFeature importances:')
    for i, column in enumerate(data_train.columns):
        print(column, clf.feature_importances_[i])
    print('\nscore_train:', score_train, '; score_val:', score_val)


if __name__ == '__main__':
    main()
