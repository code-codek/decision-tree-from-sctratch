from get_dataset import get_dataset
from tree import build_tree 
from build_tree_limited import build_tree_limited
from visualize_tree import visualize_tree

from predict import predict
from cal_acc import cal_acc

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

def main(data_file, question_dict, depth_max, plot_file):

    # Chenk inputs
    print(f'Inputs -----------------------------------------------------------------------------------')
    print(f'Dataset: {data_file}')
    print(f'Features: {question_dict}')
    print(f'Maximum tree depth: {depth_max}')
    print(f'Output file name: {plot_file}')
    print(f'-----------------------------------------------------------------------------------------')
    print()

    # Data preparation
    car_data = get_dataset(data_file)

    df_cars = pd.DataFrame(data=car_data)



    features = [value for value in question_dict.values()]
    columns = features.copy()
    columns.append("label")
    df_columns = columns

    df_cars.columns = df_columns

    print('The first 5 low of Dataframe --------------------------------------------------------------')
    print(df_cars.head())
    print('-------------------------------------------------------------------------------------------')
    print()

    X = df_cars[features]
    y = df_cars["label"]

    X = X.to_numpy()
    y = y.to_numpy()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=1)

    print(f'Length of Train data: {len(X_train)}')
    print(f' Length of Test data: {len(X_test)}')
    print()


    # Custom
    #   Train 
    acc_custom = []
    for depth in range(1, depth_max):
        # build tree
        tree = build_tree_limited(X_train,
                                  y_train,
                                  max_depth=depth)

        #   Predict
        y_pred_custom = predict(X_test, tree)
        #   Calcuate Test Acc
        acc =  cal_acc(y_pred_custom, y_test)
        acc_custom.append(acc)

    print(f'Unique values in each columns ------------------------------------------------------------')
    print(df_cars.apply(lambda col: col.unique()))
    print('-------------------------------------------------------------------------------------------')
    print()

    # sklearn
    categories = [
        ["low","med", "high","vhigh"],  # buying
        ["low","med", "high","vhigh"],  # maint
        ["2", "3", "4", "5more"],       # doors
        ["2", "4", "more"],             # persons
        ["small", "med", "big"],        # lug-boat
        ["low", "med", "high"]          # safety
    ]

    enc = OrdinalEncoder(categories = categories)

    X_train_enc = enc.fit_transform(X_train)
    X_test_enc = enc.transform(X_test)

    acc_sklearn = []
    for depth in range(1, depth_max):
        tree_clf = DecisionTreeClassifier(max_depth=depth,
                                          random_state=0)
        tree_clf.fit(X_train_enc, y_train)
        y_pred_sk = tree_clf.predict(X_test_enc)
        acc_skl = tree_clf.score(X_test_enc, y_test)
        acc_sklearn.append(acc_skl)

    print(f'acc_custom: {acc_custom}')
    print(f'acc_sklearn: {acc_sklearn}')
    print(acc_sklearn)
    plt.plot(range(1, depth_max), acc_custom, label='custom')
    plt.plot(range(1, depth_max), acc_sklearn, label='scikit learn')
    plt.xticks(np.arange(0, depth_max))
    plt.grid()
    plt.title('Max depth vs ACC in Decision Tree')
    plt.xlabel('Max depth')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(plot_file)



if __name__ == "__main__":

    data_file = './car.data'

    question_dict = {
    0: "buying",
    1: "maint",
    2: "doors",
    3: "persons",
    4: "lug_boot",
    5: "safety"
    }

    depth_max = 11

    plot_file = "./acc_vs_max_depth.png"

    main(data_file, question_dict, depth_max, plot_file)