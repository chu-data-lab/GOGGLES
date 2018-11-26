import sys

import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns

SNORKEL_LIB_DIR = '/home/goggles/snorkel'
sys.path.append(SNORKEL_LIB_DIR)
import snorkel
from snorkel.learning import GenerativeModel

from goggles.utils.labeling_matrix import make_labeling_matrix
from goggles.analysis.cross_validation import *
from goggles.utils.functional import get_performance_metrics
from goggles.context import *


def train_snorkel_gen_model(L, gte=True):
    L_train = sparse.csr_matrix(L)

    gen_model = GenerativeModel()
    gen_model.train(L_train, epochs=100, decay=0.95,
                    step_size=0.01 / L_train.shape[0],
                    reg_param=1e-6)

    train_marginals = gen_model.marginals(L_train)
    marginals_threshold = (max(train_marginals) - min(train_marginals)) / 2
    train_labels = (2 * (train_marginals >= marginals_threshold) - 1 if gte
                    else 2 * (train_marginals < marginals_threshold) - 1)

    return gen_model, train_labels, train_marginals



def train_and_validate_snorkel(model, dataset, score_thresholds, gte=True):
    L, scores, true_labels = make_labeling_matrix(
        model, dataset, score_thresholds)

    gen_model, train_labels, train_marginals = train_snorkel_gen_model(L, gte=gte)
    train_performance_metrics = get_performance_metrics(true_labels, train_labels)

    return gen_model, train_labels, train_marginals, train_performance_metrics



def eval_snorkel(gen_model, L, gte=True):
    L_test = sparse.csr_matrix(L)
    test_marginals = gen_model.marginals(L_test)
    marginals_threshold = (max(test_marginals) - min(test_marginals)) / 2
    test_labels = (2 * (test_marginals >= marginals_threshold) - 1 if gte
                   else 2 * (test_marginals < marginals_threshold) - 1)

    return test_labels, test_marginals



def train_model_with_snorkel(runs):
    training_cache = []
    train_results = []
    for i in runs:
        try:
            context = load_context_from_run_id(i)
            score_thresholds = get_best_thresholds(context.model, context.dataset.train)

            # Crossvalidation on gte and lte
            gen_model_t, train_labels_t, train_marginals_t, train_performance_metrics_t = \
                train_and_validate_snorkel(context.model, context.dataset.train, score_thresholds, gte=True)
            gen_model_f, train_labels_f, train_marginals_f, train_performance_metrics_f = \
                train_and_validate_snorkel(context.model, context.dataset.train, score_thresholds, gte=False)

            if train_performance_metrics_t.accuracy_score > train_performance_metrics_f.accuracy_score:
                cache = SimpleNamespace(
                    context=context,
                    score_thresholds=score_thresholds,
                    gen_model=gen_model_t,
                    gte=True,
                )
                results = train_performance_metrics_t.__dict__
            else:
                cache = SimpleNamespace(
                    context=context,
                    score_thresholds=score_thresholds,
                    gen_model=gen_model_f,
                    gte=False,
                )
                results = train_performance_metrics_f.__dict__

            training_cache.append(cache)

            results['Species'] = str(context.config['filter_class_ids'][0]) + ',' + str(
                context.config['filter_class_ids'][1])
            train_results.append(results)
        except FileNotFoundError:
            print("File not found for run %d", i)

    return training_cache, train_results



def test_model(training_cache):
    test_results = []
    for cache in training_cache:
        test_dataset = cache.context.dataset.test
        L, scores, true_labels = make_labeling_matrix(
            cache.context.model, test_dataset, cache.score_thresholds)
        test_labels, test_marginals = eval_snorkel(cache.gen_model, L, cache.gte)

        attribute_labels = get_attribute_labels(cache.context.model, cache.context.dataset.test)
        attribute_order = [i - 1 for i in sorted(attribute_labels, key=(lambda k: attribute_labels[k]))]
        L = (L.T[[attribute_order]]).T

        metrics = get_performance_metrics(true_labels=true_labels, predicted_labels=test_labels)
        results = metrics.__dict__
        results['Species'] = str(cache.context.config['filter_class_ids'][0]) + ',' + str(
            cache.context.config['filter_class_ids'][1])
        results['test_L'], results['test_scores'], results['true_labels'] = L, scores, true_labels
        test_results.append(results)

    return test_results

def plot_accuracy_heatmap(L):
    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(L[:len(L)//2], cmap=['#F19E7D', '#FAE8DF', '#579FC9'], linewidths=0.01,
                     linecolor='white')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(['Class 1', 'Cannot predict', 'Class 2'])
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.imshow(np.arange(9).reshape((3, 3)))
    ax.set_xlabel('Attributes', fontsize=25)
    ax.set_ylabel('Class 1 samples', fontsize=25)

    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(L[len(L)//2:], cmap=['#F19E7D', '#FAE8DF', '#579FC9'], linewidths=0.01,
                     linecolor='white')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(['Class 1', 'Cannot predict', 'Class 2'])
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.imshow(np.arange(9).reshape((3, 3)))
    ax.set_xlabel('Attributes', fontsize=25)
    ax.set_ylabel('Class 2 samples', fontsize=25)