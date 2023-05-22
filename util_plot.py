from collections import defaultdict
import sys
from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import auc, precision_recall_curve, roc_curve

from train_detector import (ClassifierDetector, CountingDetector, Detector, NeuralClassifierDetector,
                            RandomDetector, RegressorDetector)

# Map from sink to display name in plots
display_names = {
    'CommandInjection': 'Command Injection',
    'CodeInjection': 'Code Injection',
    'ReflectedXss': 'Reflected XSS',
    'TaintedPath': 'Path Traversal',
    'logging': 'Logging'
}

def get_model_name(detector: Detector):
    if isinstance(detector, NeuralClassifierDetector):
        return 'Sink Prediction'
    elif isinstance(detector, CountingDetector):
        return 'Frequency-based'
    elif isinstance(detector, ClassifierDetector):
        return 'Classifier Detector'
    elif isinstance(detector, RandomDetector):
        return 'Random Detector'
    elif isinstance(detector, RegressorDetector):
        return 'Regressor Detector'
    
def _compute_y_per_sink(ground_truth, flow_prob_dict, is_logging: bool = False) -> tuple[dict, dict]:
    # ground_truth = get_ground_truth_dict()
    y_true_per_sink = defaultdict(list)
    y_score_per_sink = defaultdict(list)
    for spec, is_unusual in ground_truth.items():
        if spec in flow_prob_dict:
            if is_logging:
              # For logging flows, a flow is unusual if it has a high probability of being in the `logging` sink
              y_score_per_sink[spec.sink].append(flow_prob_dict[spec])
            else:
              # For API flows, a flow is unusual if it has a low probability of being in the corresponding sink
              y_score_per_sink[spec.sink].append(1-flow_prob_dict[spec])
            if is_unusual:
                y_true_per_sink[spec.sink].append(1)
            else:
                y_true_per_sink[spec.sink].append(0)
    return y_true_per_sink, y_score_per_sink

def _compute_PR_curve(y_true, y_score, sink, dataset_name):
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=y_score)

    f1 = np.divide(2*precision*recall, precision+recall,
                    out=np.zeros(precision.shape, dtype=float), where=(precision+recall) != 0)
    optimal_idx = np.argmax(f1)
    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = (2*optimal_precision*optimal_recall)/(optimal_precision+optimal_recall)
    print(f"\nOptimal (best F1-score) threshold of {sink} of {dataset_name} is {1-optimal_threshold}", file=sys.stderr)
    print(f"Precision of {sink}'s optimal threshold of {dataset_name} is {optimal_precision}", file=sys.stderr)
    print(f"Recall of {sink}'s optimal threshold of {dataset_name} is {optimal_recall}", file=sys.stderr)
    print(f"F1-score of {sink}'s optimal threshold is {optimal_f1}", file=sys.stderr)

    # display = PrecisionRecallDisplay.from_predictions(
    #     y_true_per_sink[sink], y_score_per_sink[sink], name=sink)
    # _ = display.ax_.set_title(f"Precision-Recall curve of {sink}")
    return precision, recall, thresholds, optimal_f1, 1-optimal_threshold

def _get_PR_curve_per_sink(y_true_per_sink, y_score_per_sink, dataset_name):
    # PR per sink class
    precision = dict()
    recall = dict()
    thresholds = dict()
    optimal_f1 = dict()
    optimal_threshold = dict()

    for sink in sorted(y_true_per_sink.keys()):
        precision[sink], recall[sink], thresholds[sink], optimal_f1[sink], optimal_threshold[sink] = _compute_PR_curve(
                y_true_per_sink[sink], y_score_per_sink[sink], sink,
                dataset_name)
    return precision, recall, thresholds, optimal_f1, optimal_threshold

def plot_PR_curve(ground_truth, flow_prob_dict, dataset_name, model_name, is_logging: bool = False) -> dict[str, float]:
    """ Plot PR curve and return the optimal threshold """
    y_true_per_sink, y_score_per_sink = _compute_y_per_sink(ground_truth, flow_prob_dict, is_logging)
    print(dataset_name, file=sys.stderr)
    precision, recall, thresholds, optimal_f1, optimal_threshold = _get_PR_curve_per_sink(y_true_per_sink, y_score_per_sink, dataset_name)
    # print([str(f)+':'+str(p) for f,p in flow_prob_dict.items()])
    # Plot
    # last param is number of colors
    color = iter(plt.cm.rainbow(np.linspace(
        0, 1, len(precision.keys()))))
    lw = 2
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 20})
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name}\nPrecision-Recall curve of {dataset_name}")
    # Plot for each sink
    for sink in sorted(precision.keys()):
        c = next(color)
        plt.plot(
            recall[sink],
            precision[sink],
            color=c,
            lw=lw,
            label=f"{display_names[sink]} (F1-score={optimal_f1[sink]:.2f})",
        )
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'{model_name}_{dataset_name}_pr.pdf', bbox_inches='tight')
    plt.clf()
    return optimal_threshold

def _compute_ROC_curve(y_true, y_score, sink, dataset_name):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = auc(fpr, tpr)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    # print(f"\nOptimal threshold of {sink} of {dataset_name} is {1-optimal_threshold}", file=sys.stderr)
    # print(f"TPR of {sink}'s optimal threshold of {dataset_name} is {tpr[optimal_idx]}", file=sys.stderr)
    # print(f"FPR of {sink}'s optimal threshold of {dataset_name} is {fpr[optimal_idx]}", file=sys.stderr)
    return fpr, tpr, thresholds, roc_auc

def _get_ROC_curve_per_sink(y_true_per_sink, y_score_per_sink, dataset_name):
    # PR per sink class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    for sink in sorted(y_true_per_sink.keys()):
        fpr[sink], tpr[sink], thresholds[sink], roc_auc[sink] = _compute_ROC_curve(
            y_true_per_sink[sink], y_score_per_sink[sink], sink, dataset_name)
    return fpr, tpr, thresholds, roc_auc

def plot_roc_curve(ground_truth, flow_prob_dict, dataset_name, model_name, is_logging: bool = False):
    y_true_per_sink, y_score_per_sink = _compute_y_per_sink(ground_truth, flow_prob_dict, is_logging)
    fpr, tpr, thresholds, roc_auc = _get_ROC_curve_per_sink(y_true_per_sink, y_score_per_sink, dataset_name)

    # Plot
    # last param is number of colors
    color = iter(plt.cm.rainbow(np.linspace(
        0, 1, len(fpr.keys()))))
    lw = 2
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 20})
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name}\nReceiver operating characteristic of {dataset_name}")
    # Plot for each sink
    for sink in sorted(fpr.keys()):
        c = next(color)
        plt.plot(
            fpr[sink],
            tpr[sink],
            color=c,
            lw=lw,
            label=f"ROC curve of {display_names[sink]} (area = {roc_auc[sink]:.2f})",
        )
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'{model_name}_{dataset_name}_roc.pdf', bbox_inches='tight')
