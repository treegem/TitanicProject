from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.utility.data_preparation import load_clean_data
from utility.data_preparation import standard_scale


def main():
    data = load_clean_data()
    y = data.pop('Survived')
    data = standard_scale(data)
    clf = DecisionTreeClassifier()

    param_dist = {'max_depth': [3, None],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1, 5],
                  'max_features': [1, None],
                  'min_impurity_decrease': [0., 1.]}
    random_cv = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=20, n_jobs=4)
    random_cv.fit(data, y)
    print(random_cv.best_estimator_)
    # clf.fit(data_train, y_train)
    # score_train = clf.score(data_train, y_train)
    # score_val = clf.score(data_val, y_val)
    #
    # print('score_train:', score_train, '; score_val:', score_val)


if __name__ == '__main__':
    main()
