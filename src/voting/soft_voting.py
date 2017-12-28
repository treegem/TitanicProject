import numpy as np

from src.utility.data_preparation import load_clean_data, standard_scale
from src.utility.storage_utility import load_model


def main():
    models_list = load_models()

    data = load_clean_data()
    y = data.pop('Survived')
    data = standard_scale(data)

    predictions_list = predict(data, models_list)

    majority_votes = majority_vote(predictions_list, y)

    score = calc_score(majority_votes, y)

    print('Voting score:', score)


def predict(data, models_list):
    predictions_list = []
    for model in models_list:
        predictions = model.predict(data)
        predictions_list.append(predictions)
    return predictions_list


def majority_vote(predictions_list, y):
    majority_votes = np.zeros_like(y)
    for i in range(len(y)):
        ongoing_vote = 0
        for prediction in predictions_list:
            ongoing_vote += prediction[i]
        ongoing_vote /= len(predictions_list)
        majority_votes[i] = int(ongoing_vote >= 1)
    return majority_votes


def calc_score(majority_vote, y):
    correct = 0
    for i, actual_y in enumerate(y):
        correct += int(majority_vote[i] == actual_y)
    score = correct / len(y)
    return score


def load_models():  # TODO automatically load all models saved to the default model folder
    models = []
    models.append(load_model('single_tree'))
    models.append(load_model('random_forest'))
    models.append(load_model('k_nearest_neighbor'))

    return models


if __name__ == '__main__':
    main()
