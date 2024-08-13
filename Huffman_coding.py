class Node:

    def __init__(self, value: int, tag: str, parent = None):
        self.value = value
        self.tag = tag
        self.parent = parent
        self.child0 = None
        self.child1 = None
        self.id = None
        self.code = []
        return None
    

class Tree:

    def __init__(self, root: Node):
        self.root = root
        self.size = 1
        self.value = self.root.value
        self.nodes = [] #list of all nodes except of root

    def __str__(self):
        txt = ''
        for node in self.nodes:
            txt += f'node {node.id}, has code {node.code} with tag {node.tag}\n'
        return txt


    def glue_trees(self, other):
        glued_trees = Tree(Node(value = self.value + other.value, tag = '', parent = None))
        glued_trees.size += self.size + other.size
        
        glued_trees.root.child0 = self.root
        self.root.parent = glued_trees.root
        glued_trees.nodes.append(self.root)
        glued_trees.nodes.extend(self.nodes)
        nodes = self.nodes + [self.root]
        for node in nodes:
            node.code = [-1] + node.code
        
        glued_trees.root.child1 = other.root
        self.root.parent = glued_trees.root
        glued_trees.nodes.append(other.root)
        glued_trees.nodes.extend(other.nodes)
        nodes = other.nodes + [other.root]
        for node in nodes:
            node.code = [1] + node.code
        return glued_trees
    
    def update_nodes_id(self):
        for i, node in enumerate(self.nodes):
            node.id = i
            if node.tag == '':
                node.tag = i
    
    def get_node_by_tag(self, tag: str):
        for node in self.nodes:
            if node.tag == tag:
                return node
        
    def get_nodes_ids_to_node(self, node: Node):
        ids = []
        arr = node.code
        current_node = self.root
        for i in arr:
            if i == -1:
                current_node = current_node.child0
            else:
                current_node = current_node.child1
            ids.append(current_node.id)
        return ids