import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import logging
from varclr.models.model import Encoder
from util import get_ground_truth_dict, timer
from util_plot import display_names
# Ref: https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html

embeddings = Encoder.from_pretrained("varclr-codebert")

# def train_svm(train, normal, outliers):
def train_svm(train: list[str], test: list[tuple[str, bool]], sink: str):
    X_train = embeddings.encode(train)
    X_test = embeddings.encode([f for f, _ in test])
    
    # fit the model
    clf = svm.OneClassSVM(nu=1e-2, kernel="rbf", gamma=0.05, tol=1e-3, verbose=False)
    
    clf.fit(X_train)
    return clf

def get_stat(all_flows, scores: list, optimal_threshold: float):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, (name, unexpected) in enumerate(all_flows):
        if not unexpected and scores[i]>=optimal_threshold:
            fp += 1
            # print('fp', name, scores[i])
        elif unexpected and scores[i]>=optimal_threshold:
            tp += 1
            # print('tp', name, scores[i])
        elif not unexpected and scores[i]<optimal_threshold:
            tn += 1
            # print('tn', name, scores[i])
        elif unexpected and scores[i]<optimal_threshold:
            fn += 1
            # print('fn', name, scores[i])
    # double check
    # print('Positive', len([i for i in scores if i>=optimal_threshold]))
    # print('Negative', len([i for i in scores if i<optimal_threshold]))
    print('tp',tp)
    print('fp',fp)
    print('fn',fn)
    print('tn',tn)

def evaluate_dataset(spec_dict: dict, sink: str, clf, max_score: float, optimal_threshold):
    if any([spec.sink == sink for spec in spec_dict]):
        flows = [(spec.param.function+','+spec.param.parameter, unexpected) for spec, unexpected in spec_dict.items() if spec.sink == sink]
        flows_emb = embeddings.encode([f for f, _ in flows])
        scores = clf.score_samples(flows_emb)
        scores = [-float(i)/max_score for i in scores]
        get_stat(flows, scores, optimal_threshold)

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    with timer('Load ground truth'):        
        samples = {
            'CommandInjection': {
                'normal': ['execute,command'],
                'unusual': []
            },
            'CodeInjection': {
                'normal': ['eval', 'execute', 'compile', 'render', 'callback', 'function', 'fn'],
                'unusual': []
            },
            'ReflectedXss': {
                'normal': ['send,content'],
                'unusual': []
            },
            'TaintedPath': {
                'normal': ['file', 'directory', 'path', 'cwd', 'source', 'input'],
                'unusual': []
            }
        }
        ground_truth = {}
        scores = {}
        max_score = {}
        clfs = {}
        eval_datasets = {}
        # Balanced set
        eval_datasets['balanced set'] = get_ground_truth_dict()
        # Random set
        eval_datasets['random set'] = get_ground_truth_dict(ground_truth_path='data/ground-truth-full-flows.csv')
        # SecBench.js
        secbench_spec_dict = get_ground_truth_dict(ground_truth_path='data/SecBench.js.csv')
        sinks_to_test = sorted(set([spec.sink for spec in eval_datasets['balanced set'].keys()]))
        train_size_list = []
        f1_list = []
    
    for dataset_name, ground_truth_spec_dict in eval_datasets.items():
        print(f"\nEvaluating {dataset_name}:")
        for sink in sinks_to_test:
            with timer(f"Training OC-SVM for {sink}"):
                all_flows = [(spec.param.function+','+spec.param.parameter, unexpected) for spec, unexpected in ground_truth_spec_dict.items() if spec.sink == sink]
                clfs[sink] = train_svm(
                    [name for name in samples[sink]['normal']],
                    all_flows,
                    sink
                )
            with timer(f"OC-SVM predicts on ground truth for {sink}"):
                scores[sink] = clfs[sink].score_samples(embeddings.encode([f for f, _ in all_flows]))
                # normalize and flip the sign of the scores (in OneClassSVM, a low score means an outlier)
                max_score[sink] = max(scores[sink])
                scores[sink] = [-float(i)/max_score[sink] for i in scores[sink]]
                ground_truth[sink] = [1 if unexpected else 0 for _, unexpected in all_flows]

        with timer("Plotting OC-SVM"):
            plt.figure(figsize=(10, 8))
            plt.rcParams.update({'font.size': 20})
            color = iter(plt.cm.rainbow(np.linspace(
                0, 1, len(sinks_to_test))))
            # Plot ROC and AUC score
            for sink in sinks_to_test:
                c = next(color)
                y_true = ground_truth[sink]
                y_score = scores[sink]

                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, marker='.', label=f"ROC of {display_names[sink]} (AUC = {roc_auc:.2f})", color=c)
            lw = 2
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title(f"Novelty Detection\nReceiver operating characteristic of {dataset_name}")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig(f'oc_svm_{dataset_name}_roc.pdf', bbox_inches='tight')
            plt.clf()

            # Plot PR curve and F1-score
            color = iter(plt.cm.rainbow(np.linspace(
                0, 1, len(sinks_to_test))))
            for sink in sinks_to_test:
                c = next(color)
                y_true = ground_truth[sink]
                y_score = scores[sink]

                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                f1 = np.divide(2*precision*recall, precision+recall,
                                out=np.zeros(precision.shape, dtype=float), where=(precision+recall) != 0)
                optimal_idx = np.argmax(f1)
                optimal_threshold = thresholds[optimal_idx]
                optimal_precision = precision[optimal_idx]
                optimal_recall = recall[optimal_idx]
                optimal_f1 = (2*optimal_precision*optimal_recall)/(optimal_precision+optimal_recall)
                print(f"\nOptimal (best F1-score) threshold of {sink} is {optimal_threshold}")
                print(f"Precision of {sink}'s optimal threshold is {optimal_precision}")
                print(f"Recall of {sink}'s optimal threshold is {optimal_recall}")
                print(f"F1-score of {sink}'s optimal threshold is {optimal_f1}")
                plt.plot(recall, precision, marker='.', label=f"{display_names[sink]} (F1-score={optimal_f1:.2f})", color=c)
                train_size_list.append(len(samples[sink]['normal']))
                f1_list.append(optimal_f1)
                # Code for SecBench.js eval
                if dataset_name == 'balanced set':
                    print('Evaluating SecBench.js dataset:')
                    evaluate_dataset(secbench_spec_dict, sink, clfs[sink], max_score[sink], optimal_threshold)
                # Code for Debugging
                # if sink == 'CodeInjection':
                #     all_flows = [(spec.param.function+','+spec.param.parameter, unexpected) for spec, unexpected in ground_truth_spec_dict.items() if spec.sink == sink]
                #     get_stat(all_flows, scores[sink], optimal_threshold)
            lw = 2
            # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])       
            plt.title(f"Novelty Detection\nPrecision-Recall curve of {dataset_name}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig(f'oc_svm_{dataset_name}_pr.pdf', bbox_inches='tight')
        # print('train_size_list', train_size_list)
        # print('f1_list', f1_list)