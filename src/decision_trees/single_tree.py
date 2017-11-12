from sklearn.tree import DecisionTreeClassifier

from src.utility.data_preparation import load_data, train_val_split

if __name__ == '__main__':

    data = load_data()
    data_train, data_val, y_train, y_val = train_val_split()
    clf = DecisionTreeClassifier()