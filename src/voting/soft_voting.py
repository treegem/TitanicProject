from src.utility.storage_utility import load_model


def main():
    single_tree_clf, random_forest_clf, k_nearest_neighbor_clf = load_models()


def load_models():
    single_tree_clf = load_model('single_tree')
    random_forest_clf = load_model('random_forest')
    k_nearest_neighbor_clf = load_model('k_nearest_neighbor')

    return single_tree_clf, random_forest_clf, k_nearest_neighbor_clf


if __name__ == '__main__':
    main()
