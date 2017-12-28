import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utility.config import config_paths
from src.utility.data_preparation import load_clean_data


def categorize_features():
    numerical_features = ['Age', 'Fare']
    irrelevant_features = ['Name', 'Ticket', 'Cabin']
    nominal_features = ['Sex', 'Embarked']
    ordinal_features = ['Pclass', 'SibSp', 'Parch']
    return numerical_features, irrelevant_features, nominal_features, ordinal_features


def survived_countplot():
    sns.countplot(train_csv['target_name'])
    plt.xlabel('Survived?')
    plt.ylabel('#Persons')
    save_image('survived_countplot')


def correlation_matrix():
    global fig
    cor_matrix = train_csv[numerical_features + ordinal_features].corr().round(2)
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(cor_matrix, annot=True, center=0, cmap=sns.diverging_palette(250, 10, as_cmap=True),
                ax=plt.subplot(111))
    save_image('correlation_matrix')


def numerical_multiplot():
    global fig
    for column in numerical_features:
        fig = plt.figure(figsize=(18, 12))

        distribution_plot(column)
        distribution_per_survived(column)
        average_column_per_survived(column)
        boxplot_column(column)

        save_image(column)


def boxplot_column(column):
    sns.boxplot(x='target_name', y=column, data=train_csv, ax=plt.subplot(224))
    plt.xlabel('Survived?', fontsize=14)
    plt.ylabel(column, fontsize=14)


def average_column_per_survived(column):
    sns.barplot(x='target_name', y=column, data=train_csv, ax=plt.subplot(223))
    plt.xlabel('Survived?', fontsize=14)
    plt.ylabel('Average {}'.format(column), fontsize=14)


def distribution_per_survived(column):
    sns.distplot(train_csv.loc[train_csv.Survived == 0, column].dropna(),
                 color='red', label='not survived', ax=plt.subplot(222))
    sns.distplot(train_csv.loc[train_csv.Survived == 1, column].dropna(),
                 color='blue', label='survived', ax=plt.subplot(222))
    plt.legend(loc='best')
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Density per (not) survived', fontsize=14)


def distribution_plot(column):
    sns.distplot(train_csv[column].dropna(), ax=plt.subplot(221))
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.suptitle('Plots for ' + column, fontsize=18)


def ordinal_multiplot():
    global fig
    for column in ordinal_features:
        fig = plt.figure(figsize=(18, 18))
        average_per_survived(column)
        boxplot_per_survived(column)
        occurence_per_category(column)
        percentage_per_category(column)
        save_image(column)


def save_image(name):
    image_path = os.path.join(paths['images'], '{}.png'.format(name))
    image_parent_dir = os.path.dirname(image_path)
    if not os.path.isdir(image_parent_dir):
        os.makedirs(image_parent_dir)
    plt.savefig(image_path)


def percentage_per_category(column):
    sns.pointplot(x=column, y='Survived', data=train_csv, ax=plt.subplot(313))
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Survived Percentage', fontsize=14)


def occurence_per_category(column):
    ax = sns.countplot(x=column, hue='target_name', data=train_csv, ax=plt.subplot(312))
    plt.xlabel(column, fontsize=14)
    plt.ylabel('#Occurences')
    plt.legend(loc=1)
    add_percentage(ax)


def add_percentage(ax):
    # Bar heights
    height = [p.get_height() if p.get_height() == p.get_height() else 0 for p in ax.patches]
    # Counting number of bar groups
    ncol = int(len(height) / 2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width() / 2, height[i] * 1.01 + 10, '{:1.0%}'.format(height[i] / total[i]),
                ha="center", size=14)


def boxplot_per_survived(column):
    sns.boxplot(x='target_name', y=column, data=train_csv, ax=plt.subplot(322))
    plt.xlabel('Survived?', fontsize=14)
    plt.ylabel(column, fontsize=14)


def average_per_survived(column):
    sns.barplot(x='target_name', y=column, data=train_csv, ax=plt.subplot(321))
    plt.xlabel('Survived?', fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.suptitle('Plots for {}'.format(column), fontsize=18)


def nominal_multiplot():
    global fig
    for column in nominal_features:
        fig = plt.figure(figsize=(18, 12))

        survived_per_category(column)
        survived_per_category_pointplot(column)
        save_image(column)


def survived_per_category_pointplot(column):
    sns.pointplot(x=column, y='Survived', data=train_csv, ax=plt.subplot(212))
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Survived Percentage')


def survived_per_category(column):
    ax = sns.countplot(x=column, hue='target_name', data=train_csv, ax=plt.subplot(211))
    plt.xlabel(column, fontsize=14)
    plt.ylabel('#Occurences', fontsize=14)
    plt.legend(loc=1)
    plt.suptitle('Plots for {}'.format(column))
    add_percentage(ax)


def all_the_plots():
    survived_countplot()
    correlation_matrix()
    numerical_multiplot()
    ordinal_multiplot()
    nominal_multiplot()
    data = load_clean_data()
    cor_matrix = data.corr().round(2)
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(cor_matrix, annot=True, center=0, cmap=sns.diverging_palette(250, 10, as_cmap=True),
                ax=plt.subplot(111))
    save_image('full_correlation_matrix')


if __name__ == "__main__":
    paths = config_paths()
    sns.set_style('whitegrid')
    train_csv_path = os.path.join(paths['resources'], 'train.csv')
    train_csv = pd.read_csv(train_csv_path, index_col='PassengerId')
    train_csv['target_name'] = train_csv['Survived'].map({0: 'Not Survived', 1: 'Survived'})

    numerical_features, irrelevant_features, nominal_features, ordinal_features = categorize_features()

    all_the_plots()
