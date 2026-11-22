from collections import Counter
from tree import find_best_split, split
from tree import Leaf, Internal_Node

def build_tree_limited(data, 
                       labels, 
                       value="",
                       depth=0,
                       max_depth=3
                       ):
    
    best_feature, best_gain = find_best_split(data, 
                                              labels)

    if depth >= max_depth:
        return Leaf(Counter(labels), value)

    if best_gain == 0:
        leaf = Leaf(Counter(labels), value)
        return leaf

    # 最も情報利得が大きい値になる分け方でのサブセットとラベル
    data_subsets, label_subsets = split(data, 
                                        labels, 
                                        best_feature) 

    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree_limited(data_subsets[i],
                            label_subsets[i],
                            data_subsets[i][0][best_feature],
                            depth + 1,
                            max_depth)

        branches.append(branch)

    return Internal_Node(best_feature,
                         branches,
                         value,
                         len(labels),
                         labels)