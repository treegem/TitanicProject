from sklearn.model_selection import RandomizedSearchCV


def randomized_search_cv(clf, data, parameter_distribution, targets,
                         n_iter=10000, n_jobs=1, cv=3, verbose=False):
    random_cv = RandomizedSearchCV(estimator=clf,
                                   param_distributions=parameter_distribution,
                                   n_iter=n_iter, n_jobs=n_jobs,
                                   cv=cv, verbose=verbose)
    # n_jobs = number of parallel threads, should not exceed number of cores
    random_cv.fit(data, targets)
    print(random_cv.best_estimator_)
    print(random_cv.best_score_)
    print(random_cv.best_params_)
    clf = random_cv.best_estimator_
    return clf
