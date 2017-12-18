from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier

from src.utility.data_preparation import split_data, load_clean_data, standard_scale
from src.utility.parameter_search import randomized_search_cv
from src.utility.storage_utility import save_model


def main():
    data = load_clean_data()
    y = data.pop('Survived')
    data = standard_scale(data)
    clf = RandomForestClassifier()

    param_dist = {'n_estimators': randint(low=50, high=500),
                  'max_depth': list(range(1, 20)) + [None],
                  'min_samples_split': list(range(2, 10)),
                  'min_samples_leaf': list(range(1, 10)),
                  'max_features': list(range(1, 10)) + [None],
                  'min_impurity_decrease': uniform(loc=0, scale=1)}
    clf = randomized_search_cv(clf, data, parameter_distribution=param_dist,
                               targets=y, n_jobs=3, n_iter=50, cv=3, verbose=True)

    data = load_clean_data()
    data_train, data_val, y_train, y_val = split_data(data)
    data_train = standard_scale(data_train)
    data_val = standard_scale(data_val)
    # clf.fit(data_train, y_train)
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    save_model(clf, file_name='random_forest')

    print('\nFeature importances:')
    for i, column in enumerate(data_train.columns):
        print(column, clf.feature_importances_[i])
    print('score_train:', score_train, '; score_val:', score_val)


if __name__ == "__main__":
    main()
