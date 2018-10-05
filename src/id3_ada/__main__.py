import argparse, datetime, math, warnings
import pandas as pd
import sklearn.metrics as sklmetrics
from sklearn.preprocessing import LabelEncoder

from id3_ada.random_forest import RandomForestClassifier
from id3_ada.all_features_encoder import AllFeaturesLabelEncoder

def confusion_matrix(true_labels, predicted_labels):
    sorted_labels = sorted(list( set(true_labels) | set(predicted_labels) ))
    conf_mat = sklmetrics.confusion_matrix(true_labels, predicted_labels,
            labels = sorted_labels)
    conf_mat = pd.DataFrame(conf_mat, 
            columns = pd.Index(sorted_labels, name='Predicted'), 
            index = pd.Index(sorted_labels, name = 'True'))
    return conf_mat

def show_performance_stats(true_labels, predicted_labels):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(true_labels, predicted_labels))
    
    print('\nClassification Report:\n')
    print(sklmetrics.classification_report(true_labels, predicted_labels))
    print('Accuracy: {:.4f}'.format(sklmetrics.accuracy_score(true_labels, predicted_labels)))

def show_oob_statistics(classifier):
    print('Out-of-Bag Accuracy: {:.4f}'.format(classifier.oob_accuracy()))

    feature_importances = pd.DataFrame(list(classifier.feature_importances().items()),
            columns = ['Feature', '% Decrease in Accuracy'])
    feature_importances.loc[:, '% Decrease in Accuracy'] = (feature_importances.loc[:, '% Decrease in Accuracy'] * 100)
    feature_importances.sort_values('% Decrease in Accuracy', ascending = False, inplace = True)
    feature_importances.reset_index(drop = True, inplace = True)
    print('\nFeature Importances:\n', feature_importances)

def omit_invalid_rows(data, labels, feature_encoder, label_encoder):
    """ Remove rows in data and labels which contain feature values that were 
        not seen during training """
    is_valid = feature_encoder.is_valid(data)
    is_valid &= labels.isin(label_encoder.classes_)

    return (data.loc[is_valid, :], labels[is_valid])

def load_data(path, target_col):
    data = pd.read_csv(path, index_col = False, dtype = 'object')
    labels = data.pop(target_col)
    return (data, labels)

def train_model_and_print_summary(train_data_file, target_col, print_tree = False, 
            test_data_file = None, ntree = 100):
    
    num_trees = ntree
    train_data, raw_train_labels = load_data(train_data_file, target_col)
    num_attrs = len(train_data.columns)
    num_rand_split_attrs = min(num_attrs, math.floor(math.sqrt(num_attrs)))

    # Preprocessing
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(raw_train_labels)
    feature_encoder = AllFeaturesLabelEncoder().fit(train_data)
    train_data = feature_encoder.transform(train_data)

    # Build classifier
    classifier = RandomForestClassifier(num_rand_split_attrs, num_trees, 
            len(label_encoder.classes_), compute_stats = True)

    print('\nPARAMETERS')
    print('\tNumber of trees:', num_trees)
    print('\tMaximum number of random features to try per split:', num_rand_split_attrs)

    print('\nBulding Random Forest. This may take a while for large datasets or for large numbers of trees.')
    t0 = datetime.datetime.now()
    classifier.fit(train_data, train_labels)
    print('\nDone building forest (Completed in: {})'.format(datetime.datetime.now() - t0))

    # Show performance stats

    print('\n\nOUT-OF-BAG STATISTICS\n-----------')
    show_oob_statistics(classifier)

    print('\n\nTRAINING SET PERFORMANCE STATISTICS\n-----------')
    # Use unencoded values so output is meaninful for user
    train_predictions = label_encoder.inverse_transform(classifier.predict(train_data))
    show_performance_stats(raw_train_labels, train_predictions)

    if not test_data_file is None:
        raw_test_data, raw_test_labels = load_data(test_data_file, target_col)

        # TODO: Print warning when rows are ommitted
        raw_test_data, raw_test_labels = omit_invalid_rows(raw_test_data, raw_test_labels, 
                feature_encoder, label_encoder)
        
        #test_labels = label_encoder.transform(raw_test_labels)
        test_data = feature_encoder.transform(raw_test_data)

        print('\n\nTEST SET PERFORMANCE STATISTICS\n-----------')
        # Use unencoded values so output is meaninful for user
        test_predictions = label_encoder.inverse_transform(classifier.predict(test_data))
        show_performance_stats(raw_test_labels, test_predictions)

def main():

    # Suppress sci-kit learn depreciation warnings
    warnings.simplefilter('ignore', category = DeprecationWarning)

    parser = argparse.ArgumentParser(
        prog = 'id3_ada', 
        description = 'Trains a Random Forest of ID3 decision trees on the specified dataset using all fields except for <target> as predictors. All fields are assumed to be categorical.',
        epilog = 'Author: Jordan Finch (jordan.finch@student.uts.edu.au)'
    )
    parser.add_argument(
        'train_data_file',
        metavar = 'trainfile',
        help = 'Path to the training data. Must be a csv file'
    )
    parser.add_argument(
        'target_col',
        metavar = 'target',
        help = 'Name of the column to classify.'
    )
    parser.add_argument(
        '--test-data-file',
        help = 'Path to the test data. Must be a csv file'
    )
    parser.add_argument(
        '--ntree',
        default = '100',
        type = int,
        help = 'Number of trees to grow'
    )
    args = parser.parse_args()
    
    train_model_and_print_summary(**vars(args))


if __name__ == '__main__':
    main()