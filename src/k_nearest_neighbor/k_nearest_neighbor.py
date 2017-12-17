from scipy.stats import uniform
from sklearn.neighbors import KNeighborsClassifier

from src.utility.data_preparation import load_clean_data, standard_scale, split_data
from src.utility.parameter_search import randomized_search_cv
from src.utility.storage_utility import save_model


def main():
    data = load_clean_data()
    y = data.pop('Survived')
    data = standard_scale(data)
    clf = KNeighborsClassifier()

    param_dist = {'n_neighbors': list(range(1, 50)),
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'p': uniform(loc=1, scale=3)}
    clf = randomized_search_cv(clf, data, parameter_distribution=param_dist,
                               targets=y, n_jobs=3, n_iter=10, cv=4, verbose=True)

    data = load_clean_data()
    data_train, data_val, y_train, y_val = split_data(data)
    data_train = standard_scale(data_train)
    data_val = standard_scale(data_val)
    score_train = clf.score(data_train, y_train)
    score_val = clf.score(data_val, y_val)

    save_model(clf, file_name='k_nearest_neighbor')

    print('\nscore_train:', score_train, '; score_val:', score_val)


if __name__ == '__main__':
    main()
