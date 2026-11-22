from tree import Leaf
from tree import Internal_Node
import operator

def classify(datapoint, tree):
    """
    入力datapoinyをtreeに従って分類する
    
    datapoint;
        分類予測に用いるデータポイント
    
    tree;
        作成された決定木
    """

    if isinstance(tree, Leaf):
        return max(tree.labels.items(),
                   key=operator.itemgetter(1))[0]

    # treeの分離に使用したfeatureの値を取得
    value = datapoint[tree.feature]

    # 各branchにおける該当するfeatureの値と
    # datapointのfeatureの値が等くなる分岐を探す
    for branch in tree.branches:
        if branch.value == value:
            # 等しい場合、次の分類に入る
            return classify(datapoint, branch)

    # 分岐する枝がない場合には、そのノードの多数派を返す
    label_majority =  max(tree.labels, key=tree.labels.get)
    return label_majority