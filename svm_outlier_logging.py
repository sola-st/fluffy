from collections import namedtuple
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from varclr.models.model import Encoder
# Ref: https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html

embeddings = Encoder.from_pretrained("varclr-codebert")

# def train_svm(train, normal, outliers):
def train_svm(train: list[str], test: list[tuple[str, bool]], sink: str):
    # print(train)
    # print(test[:5])
    X_train = embeddings.encode(train)
    # X_test = embeddings.encode([f for f, _ in test])
    # fit the model
    clf = svm.OneClassSVM(nu=1e-2, kernel="rbf", gamma=0.05, tol=1e-3, verbose=False)
    clf.fit(X_train)
    return clf

def read_logging_flows(path: str):
    flows_dict = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            flow = namedtuple('Flow', ['name', 'sink', 'label'])
            flow.name = row[0]
            flow.sink = row[1]
            flow.label = row[2]
            flows_dict[flow] = flow.label
    return flows_dict

def get_stat(all_flows, scores: list, optimal_threshold: float):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, (name, label) in enumerate(all_flows):
        if (label == 'FP' or label == 'TN') and scores[i]>=optimal_threshold:
            fp += 1
            # print('fp', name)
            # print(scores[i]) 
        elif (label == 'TP' or label == 'FN') and scores[i]>=optimal_threshold:
            tp += 1
            # print('tp', name)
            # print(scores[i]) 
        elif (label == 'FP' or label == 'TN') and scores[i]<optimal_threshold:
            tn += 1
            # print('tn', name)
            # print(scores[i]) 
        elif (label == 'TP' or label == 'FN') and scores[i]<optimal_threshold:
            fn += 1
            # print('fn', name)
            # print(scores[i]) 
    # double check
    # print('Positive', len([i for i in scores if i>=optimal_threshold]))
    # print('Negative', len([i for i in scores if i<optimal_threshold]))
    print('tp',tp)
    print('fp',fp)
    print('fn',fn)
    print('tn',tn)

if __name__ == '__main__':
    samples = {
        'logging': {
            'normal': ['authkey', 'password', 'passcode', 'passphrase'],
            'unusual': []
        },        
    }
    ground_truth = {}
    scores = {}
    # ground_truth_spec_dict = {}
    # not_sensitive_spec_dict = {}
    dataset_total_path = 'data/logging_flows_ground_truth.csv'
    dataset_not_sensitive_path = 'data/logging_flows_not_sensitive_unique.csv'
    all_flows_spec_dict = read_logging_flows(dataset_total_path)
    not_sensitive_spec_dict = read_logging_flows(dataset_not_sensitive_path)
    
    sinks_to_test = sorted(set([spec.sink for spec in all_flows_spec_dict.keys()]))
    train_size_list = []
    f1_list = []
    loop_for_self_label = 1
    for i in range(loop_for_self_label):
        for sink in sinks_to_test:
            all_flows = [(spec.name, label) for spec, label in all_flows_spec_dict.items() if spec.sink == sink]
            clf = train_svm(
                [name for name in samples[sink]['normal']],
                all_flows,
                sink
            )
            all_flows_emb = embeddings.encode([f for f, _ in all_flows])
            scores[sink] = clf.score_samples(all_flows_emb)

            ## No need to flip, as we want to catch the "normal" cases (i.e. words similar to "password", etc.)
            # normalize and flip the sign of the scores (in OneClassSVM, a low score means an outlier)
            # scores[sink] = [-float(i)/max(scores[sink]) for i in scores[sink]]
            
            ground_truth[sink] = [1 if label == 'TP' or label == 'FN' else 0 for _, label in all_flows]

        colors = {
            'logging': 'red',
            'NoSkill': 'blue'
        } 
        # Plot ROC and AUC score
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': 20})
        color = iter(plt.cm.rainbow(np.linspace(
            0, 1, len(sinks_to_test))))
        plt.plot([0, 1], [0, 1], linestyle='--', label='NoSkill', color=colors['NoSkill'])
        for sink in sinks_to_test:
            c = next(color)
            y_true = ground_truth[sink]
            y_score = scores[sink]

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, marker='.', label=f"ROC of {sink} (AUC = {roc_auc:.2f})", color=c)
        lw = 2
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f"Novelty Detection\nReceiver operating characteristic of logging")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('oc_svm_logging_roc.pdf', bbox_inches='tight')
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
            plt.plot(recall, precision, marker='.', label=f"{sink} (F1-score={optimal_f1:.2f})", color=c)
            train_size_list.append(len(samples[sink]['normal']))
            f1_list.append(optimal_f1)

            # Find FNs in non-sensitive flows
            # not_sensitive_flows = [(spec.name, label) for spec, label in not_sensitive_spec_dict.items() if spec.sink == sink]
            # not_sensitive_flows_emb = embeddings.encode([f for f, _ in not_sensitive_flows])
            # not_sensitive_flows_score = clf.score_samples(not_sensitive_flows_emb)
            # get_stat(not_sensitive_flows, not_sensitive_flows_score, optimal_threshold)

            # Code for Debugging
            if sink == 'logging':
                all_flows = [(spec.name, label) for spec, label in all_flows_spec_dict.items() if spec.sink == sink]
                get_stat(all_flows, scores[sink], optimal_threshold)
       
        lw = 2
        # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])       
        plt.title(f"Novelty Detection\nPrecision-Recall curve of logging")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('oc_svm_logging_pr.pdf', bbox_inches='tight')

    # print('train_size_list', train_size_list)
    # print('f1_list', f1_list)