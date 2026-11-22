from get_dataset import get_dataset
from tree import build_tree 
from build_tree_limited import build_tree_limited
from visualize_tree import visualize_tree

from predict import predict
from cal_acc import cal_acc

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

def main(data_file, question_dict, max_depth):
    # Chenk inputs
    print(f'Inputs -----------------------------------------------------------------------------------')
    print(f'Dataset: {data_file}')
    print(f'Features: {question_dict}')
    print(f'Maximum tree depth: {max_depth}')
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
    tree = build_tree_limited(X_train,
                              y_train,
                              max_depth=max_depth)

    #   Predict labels using training data
    pred_train_labels = predict(X_train, tree)
    #   Calcuate ACC using training data
    acc_custom_train = cal_acc(pred_train_labels, y_train)
    print(f'ACC using training data: {acc_custom_train}')
    print()
    #   Visualize tree
    print('Rendeer the tree')
    print()
    dot = visualize_tree(tree, question_dict)
    # print(type(dot))
    # print(dot)
          

    # dot.render("deccision_tree", 
    dot.render("trained_decision_tree_limited", 
            #    format="png", 
               format="svg", 
               view=False)

    # y_pred_custom = predict(X_test, tree)

    #   Predict
    y_pred_custom = predict(X_test, tree)

    #   Calcuate Test Acc
    acc =  cal_acc(y_pred_custom, y_test)
    # print(f'ACC: {acc}')

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

    tree_clf = DecisionTreeClassifier(max_depth=max_depth,
                                      random_state=0)

    tree_clf.fit(X_train_enc, y_train)

    y_pred_sk = tree_clf.predict(X_test_enc)

    sk_score = tree_clf.score(X_test_enc, y_test)

    print()
    print(f'-------------------------------------------------------------------')
    print(f'Test ACC Score')
    print(f'custom: {acc}, sklearn: {sk_score}')
    print(f'-------------------------------------------------------------------')

    comparison = pd.DataFrame({
        "actual": y_test,
        "custom": y_pred_custom,
        "sklearn": y_pred_sk
    })
    comparison["same_prediction"] =(
        comparison["custom"] == comparison["sklearn"]
    )
    print()
    count = ((comparison["actual"] == comparison["custom"]) & (comparison["actual"] != comparison["sklearn"])).sum()
    print(f'Number of custom only correct: {count}')

    count = ((comparison["actual"] == comparison["sklearn"]) & (comparison["actual"] != comparison["custom"])).sum()
    print(f'Number of scikit-learn only correct: {count}')

    count = ((comparison["actual"] != comparison["sklearn"]) &
              (comparison["actual"] != comparison["custom"]) & 
              (comparison["sklearn"] != comparison["custom"])).sum()
    print(f'Number of both incorrect (different wrong labels): {count}')
    print()

    print(comparison)

    diff = comparison[comparison["same_prediction"] == False]

    print(f'Number of different predictions: {len(diff)}')
    print(diff)
    # print(diff.index)

    print()
    print(f"Feature data for witch custom predictions and scikit-learn predictions differ")
    for index in diff.index:
        print(index, X_test[index])

    print()
    print(f'Maximum tree depth: {max_depth}')

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

    max_depth = 3

    main(data_file, question_dict, max_depth)