import collections
import numpy as np
import pandas as pd

# From this package
from id3_ada import metrics

class ID3Node:
    """ Not intended for use outside of ID3Tree """

    def __init__(self, depth = None, entropy = None):
        self._depth = depth
        self._entropy = entropy
        self._is_leaf = False
        self._label = None
        self._children = dict()
        self._split_attr = None
    
    def get_label(self):
        return self._label

    def get_depth(self):
        return self._depth

    def set_to_leaf(self, label):
        self._is_leaf = True
        self._label = label

    def is_leaf(self):
        return self._is_leaf

    def get_split_attr(self):
        return self._split_attr

    def set_split_attr(self, attr):
        self._is_leaf = False
        self._split_attr = attr

    def add_child(self, split_attr_val, node):
        self._children[split_attr_val] = node

    def get_child_for_instance(self, data_instance: collections.namedtuple):
        try:
            return self._children[data_instance[self._split_attr]]
        except KeyError:
            print('Found unknown value "{}" in column {}'.format(data_instance[self._split_attr], self._split_attr))
            raise

    def get_children(self):
        return self._children.items()


class ID3Tree:

    # Flags
    NONE_INCREASE_INFO_FLAG = '___NONE_INCREASE_INFO___'

    def __init__(self, num_rand_split_attrs):
        self._num_rand_split_attrs = num_rand_split_attrs
        # Seting during fitting
        self._root = None
        self._category_levels = None
    
    def fit(self, data: pd.DataFrame, labels: np.ndarray, category_levels: dict):
        """ Iterative implementation of the ID3 algorithm
        Args:
            data: DataFrame containing features
            labels: pd.Series or np.ndarray containing target labels
            category_levels: Dict keyed by feature name with values containing the set of values that each feature can take
        """
        self._category_levels = category_levels

        unprocessed_nodes = collections.deque()
            
        # Init root node (split attribute is set later)
        self._root = ID3Node(depth = 0, entropy = metrics.entropy(labels))
        unprocessed_nodes.append( ( self._root, set(data.columns), 
                pd.Series([True] * len(labels)) ) )
        
        while len(unprocessed_nodes) > 0:
            current_node, current_candidate_attrs, current_instance_mask = unprocessed_nodes.popleft()
            current_labels = labels[current_instance_mask]

            if len(current_candidate_attrs) == 0 or len(np.unique(current_labels)) == 1:
                # If there are no candidate attributes or all instances belong to one class
                current_node.set_to_leaf(self._mode(current_labels))
            else:
                # Find next attr to split on
                current_data = data.loc[current_instance_mask, current_candidate_attrs]
                split_attr = self._next_split_attribute(current_data, current_labels)

                if split_attr is ID3Tree.NONE_INCREASE_INFO_FLAG:
                    # If no unused attributes increase information
                    current_node.set_to_leaf(self._mode(current_labels))
                    continue
                
                current_node.set_split_attr(split_attr)

                child_candidate_attrs = current_candidate_attrs.difference([split_attr])
                for value in self._category_levels[split_attr]:
                    
                    child_instance_mask = current_instance_mask & \
                            (data.loc[current_instance_mask, split_attr] == value)
                    
                    child_node = ID3Node(depth = current_node.get_depth() + 1,
                            entropy = metrics.entropy(labels[child_instance_mask]))
                    
                    current_node.add_child(value, child_node)

                    if len(data.loc[child_instance_mask, :]) == 0:
                        child_node.set_to_leaf(self._mode(current_labels))
                    else:
                        unprocessed_nodes.append((child_node, child_candidate_attrs, child_instance_mask))
    
    def predict(self, data: pd.DataFrame):
        return data.apply(lambda instance: self.predict_instance(instance), axis = 1)

    def show_tree(self):
        """ Prints a text representation of the decision tree.

        Implementation Note: Performs a pre-order depth-first traversal 
            of the tree, printing details of each node when visited
        """
        # Treating this list like a stack
        nodes = [(None, None, self._root)]
        
        while len(nodes) > 0:
            parent_split_attr, parent_split_val, node = nodes.pop()

            padding_text = ''
            if not node is self._root:
                padding_text = '{} {}={} '.format('-' * node.get_depth(), parent_split_attr, parent_split_val)
            
            if node.is_leaf():
                print(padding_text + '(predicted class: {}, entropy: {:.4f})'.format(node.get_label(), node._entropy))
            else:

                for split_val, child in node.get_children():
                    nodes.append((node.get_split_attr(), split_val, child))

                if node is self._root:
                    print('Root (splits on: {}, entropy: {:.4f})'.format(node.get_split_attr(), node._entropy))
                else:
                    print(padding_text + '(splits on: {}, entropy: {:.4f})'.format(node.get_split_attr(), node._entropy))
    
    def predict_instance(self, data_instance: collections.namedtuple):
        current = self._root
        while not current.is_leaf():
            current = current.get_child_for_instance(data_instance)
        return current.get_label()
    
    def _mode(self, x):
        """Returns the mode (most frequent) value of x"""
        value_counts = np.unique(x, return_counts = True)
        # Most frequent value
        mode = value_counts[0][np.argmax(value_counts[1])]
        return mode
    
    def _random_attr_subset(self, attrs):
        if len(attrs) > self._num_rand_split_attrs:
            return np.random.choice(attrs, self._num_rand_split_attrs,
                    replace = False)
        return attrs
    
    def _next_split_attribute(self, data: pd.DataFrame, labels: pd.Series):
        data = data.loc[:, self._random_attr_subset(data.columns.values)]
        info_gains = data.apply(lambda x: self._info_gain(x, labels), axis = 0)
        max_gain = info_gains.max()

        max_gain_attrs = (info_gains[info_gains == max_gain]).index.values
        
        # If no attributes increase information, no next split attribute
        if (info_gains == 0).all():
            return ID3Tree.NONE_INCREASE_INFO_FLAG
        
        # If there are ties, select one attr randomly
        if len(max_gain_attrs) > 1:
            return np.random.choice(max_gain_attrs, 1)[0]
        
        return max_gain_attrs[0]

    def _info_gain(self, x: pd.Series, labels: pd.Series):
        # Has format (unique_values, frequencies)
        value_counts = np.unique(x, return_counts = True)
        num_obs = len(labels)
        
        weighted_sub_entropy_sum = 0
        for value, freq in zip(value_counts[0], value_counts[1]):
            sub_entropy = metrics.entropy(labels[x == value])
            weighted_sub_entropy_sum += (freq / num_obs) * sub_entropy

        complete_entropy = metrics.entropy(labels)
        return complete_entropy - weighted_sub_entropy_sum

