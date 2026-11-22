from classify import classify

def predict(X_test, tree):
    pred_labels = []
    for i in range(len(X_test)):
        pred_label = classify(X_test[i], tree)
        pred_labels.append(pred_label) 

    return pred_labels