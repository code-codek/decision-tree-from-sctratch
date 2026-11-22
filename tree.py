from collections import Counter

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, labels_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, labels_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def split(dataset, labels, column):
    """
        datasetを分離する。
        column列のlabelによって、subsetに分けて返す。
    """
    data_subsets = []
    label_subsets = []
    # dataset内のcolumn列の全種のデータから、ラベルを抽出
    counts = list(set([data[column] for data in dataset])) # type class set
    counts.sort()

    # columnで選択されたラベルについて、
    # 同一ラベルごとのセットに分離する
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)

    return data_subsets, label_subsets

def gini(dataset):
    impurity = 1
    label_counts = Counter(dataset)
    for label in label_counts:
        prob_of_label = label_counts[label] / len(dataset)
        impurity -= prob_of_label ** 2

    return impurity

def information_gain(staring_labels, split_labels):
    info_gain = gini(staring_labels)
    for subset in split_labels:
        # 重み付きジニ係数を引く
        info_gain -= gini(subset) * len(subset) / len(staring_labels)

    return info_gain

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.samples = sum(self.labels.values())
        self.value = value

    def __str__(self):
        return f"""
Class Leaf
labels: {self.labels}
 value: {self.value}"""

class Internal_Node:
    def __init__(self,
                feature,
                branches,
                value,
                samples,
                labels):
        self.feature = feature
        self.branches = branches
        self.value = value
        self.samples = samples
        self.labels = Counter(labels)

    def __str__(self):
        return f"""
Class Internal_Node:
 feature: {self.feature}
branches: {self.branches}
   value: {self.value}
  labels: {self.labels}"""


def build_tree(data, labels, value=""):
    best_feature, best_gain = find_best_split(data, 
                                              labels)

    if best_gain == 0:
        leaf = Leaf(Counter(labels), value)
        return leaf

    # 最も情報利得が大きい値になる分け方でのサブセットとラベル
    data_subsets, label_subsets = split(data, 
                                        labels, 
                                        best_feature) 

    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i],
                            label_subsets[i],
                            data_subsets[i][0][best_feature])

        branches.append(branch)

    return Internal_Node(best_feature,
                         branches,
                         value,
                         len(labels),
                         labels)