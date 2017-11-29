from sklearn.model_selection import RandomizedSearchCV


def randomized_search_cv(clf, data, param_dist, targets, n_iter=10000, n_jobs=1):
    random_cv = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=n_iter, n_jobs=n_jobs)
    random_cv.fit(data, targets)
    print(random_cv.best_estimator_)
    print(random_cv.best_score_)
    print(random_cv.best_params_)
    clf = random_cv.best_estimator_
    return clf