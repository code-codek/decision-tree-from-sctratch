from graphviz import Digraph

from tree import Leaf, Internal_Node

def visualize_tree(node, question_dict, dot=None, parent_id=None):
    """
    parent_id
        親がない場合: None
        親がある場合: 親のid
    """

    if dot is None:
        dot = Digraph()
        dot.attr("node", shape="box")
    # オブジェクトがメモリ上のアドレスを
    # 整数で表現したもの
    node_id = str(id(node))

    if isinstance(node, Leaf):
        label = f"Leaf\n{node.labels}\nsamples: {node.samples}"
    else:
        label = f"Split by:\n{question_dict[node.feature]}\nsamples: {node.samples}"

    dot.node(node_id, label)

    if parent_id is not None:
        dot.edge(parent_id, node_id, label=str(node.value))

    if isinstance(node, Internal_Node):
        for branch in node.branches:
            visualize_tree(branch, question_dict, dot, node_id)

    return dot