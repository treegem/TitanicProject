from scipy.stats import uniform
from sklearn.tree import DecisionTreeClassifier

from src.utility.data_preparation import load_clean_data
from src.utility.data_preparation import standard_scale
from src.utility.parameter_search import randomized_search_cv
from src.utility.storage_utility import save_model


def main():
    data = load_clean_data()
    y = data.pop('Survived')
    data = standard_scale(data)
    n_features = len(data.columns)
    clf = DecisionTreeClassifier()

    param_dist = {'max_depth': list(range(1, 50)) + [None],
                  'min_samples_split': list(range(2, 10)),
                  'min_samples_leaf': list(range(1, 10)),
                  'max_features': list(range(1, n_features)) + [None],
                  'min_impurity_decrease': uniform(loc=0, scale=1)}
    clf = randomized_search_cv(clf, data, parameter_distribution=param_dist,
                               targets=y, n_jobs=3, n_iter=1000, cv=5,
                               verbose=True)

    score = clf.score(data, y)
    save_model(clf, file_name='single_tree')

    print('\nFeature importances:')
    for i, column in enumerate(data.columns):
        print(column, clf.feature_importances_[i])
    print('\nscore:', score)


if __name__ == '__main__':
    main()
