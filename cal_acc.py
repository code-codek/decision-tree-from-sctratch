def cal_acc(pred_labels, y_test):
    count = 0
    for i in range(len(y_test)):
        if pred_labels[i] == y_test[i]:
            count += 1

    return count / len(y_test)