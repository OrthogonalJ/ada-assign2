import collections, math
import numpy as np
import pandas as pd

# From this package
from id3_ada import id3

class DotLogger:

    def __init__(self, new_line_every = 100):
        self._i = 0
        self._new_line_every = new_line_every

    def __call__(self):
        self._i += 1
        print('.', end = '', flush = True)
        if self._i % self._new_line_every == 0:
            print('')
            self._i = 0

class RandomForestClassifier:

    def __init__(self, num_rand_split_attrs, num_trees, num_classes, 
                compute_stats = False):
        self._num_rand_split_attrs = num_rand_split_attrs
        self._num_trees = num_trees
        self._num_classes = num_classes
        self._trees = []
        self._compute_stats = compute_stats
        self._is_fit = False

        # Init optional attributes
        self._reset_stats_if_enabled()
    
    def fit(self, data: pd.DataFrame, labels: np.ndarray):
        self._is_fit = True
        # Reset
        self._trees = []
        self._reset_stats_if_enabled()
        dot_logger = DotLogger()

        self._category_levels = {attr: set(data.loc[:, attr]) for attr in data.columns.values}
        
        if self._compute_stats:
            self._train_data = data
            self._train_labels = labels

        for _ in range(self._num_trees):
            dot_logger()
            bag_index = self._bag_sample(len(data))

            if self._compute_stats:
                self._bag_instance_sets.append(bag_index)

            tree = id3.ID3Tree(self._num_rand_split_attrs)
            tree.fit(
                data.loc[bag_index, :] \
                    .reset_index(drop = True),
                labels[bag_index],
                self._category_levels
            )
            self._trees.append(tree)

    def predict(self, data: pd.DataFrame):
        """ Predict batch of instances """
        return data.apply(lambda data_instance: self.predict_instance(data_instance), 
                axis = 1)

    def predict_instance(self, data_instance: collections.namedtuple):
        """ Prediction one instance """
        tree_preds = [tree.predict_instance(data_instance) for tree in self._trees]
        vote_counts = np.unique(tree_preds, return_counts = True)
        max_vote_preds = vote_counts[0][vote_counts[1] == vote_counts[1].max()]

        # If there are ties, select one label randomly
        if len(max_vote_preds) > 1:
            return np.random.choice(max_vote_preds, 1)[0]
        
        return max_vote_preds[0]

    def feature_importances(self):
        """ Returns dict keyed by feature name with values containing the 
            decrease in oob accuracy obsevered after random permutation.
        """
        self._check_stats_enabled('feature_importances')
        feature_importances = {}
        oob_accuracy = self.oob_accuracy()
        num_instances = len(self._train_data)
        
        for col, levels in self._category_levels.items():
            noisy_data = self._train_data.copy()
            noisy_data.loc[:, col] = np.random.choice(list(levels), num_instances)
            feature_importances[col] = oob_accuracy - (self._oob_predictions(noisy_data) == self._train_labels).mean()

        return feature_importances
    
    def oob_accuracy(self):
        """ Returns accuracy using out-of-bag instances """
        self._check_stats_enabled('oob_accuracy')

        # Caching the out-of-bag accuracy because it is expensive to compute
        # NOTE: Cache is set to None when the model is refit
        if self._oob_accuracy_cached is None:
            self._oob_accuracy_cached = (self._train_labels == self._oob_predictions()).mean()
        
        return self._oob_accuracy_cached
    
    def _oob_predictions(self, train_data: pd.DataFrame = None):
        """
        Args: 
            train_data: modified version training data passed to fit. 
                This method uses self._train_data if not provided.
        
        Returns: ndarray of predicted classes for each instance in train_data 
            based trees for which each instances was out-of-bag
        """
        train_data = train_data if not train_data is None else self._train_data
        self._check_stats_enabled('_oob_predictions')
        
        vote_counts = np.zeros((len(self._train_data), self._num_classes), dtype = np.int16)
        all_instances = set(range(len(self._train_data)))

        for bag_indices, tree in zip(self._bag_instance_sets, self._trees):
            oob_indices = all_instances.difference(bag_indices)
            tree_predictions = tree.predict(train_data.loc[oob_indices, :])

            for i in oob_indices:
                vote_counts[i, tree_predictions[i]] += 1
        
        return np.argmax(vote_counts, axis = 1)

    def _check_stats_enabled(self, caller):
        """ Check if Out-of-bag computation has been requested without being enabled """
        if not self._compute_stats:
            raise ValueError("Must set compute_stats=True when constructing RandomForestClassifier before calling {}".format(caller))
        elif not self._is_fit:
            raise ValueError("Must fit RandomForestClassifier before calling {}".format(caller))

    def _reset_stats_if_enabled(self):
        """ Re-inits properties required to OOB statistics """
        if self._compute_stats:
            self._bag_instance_sets = []
            self._train_data = None
            self._train_labels = None
            self._oob_accuracy_cached = None

    def _bag_sample(self, num_instances):
        """ Samples a new bag and returns a bool mask indicating which rows are in it. """
        return np.random.choice(np.arange(num_instances), num_instances, True)
