import os
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable, Optional, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchmetrics import F1Score
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from varclr.models.model import Encoder

from classifier_net import ClassifierNet
from flows import FlowSpec, FlowSpecs, ParamRepr, ParamReprSinkPair, SinkKind
from flows_dataset import FlowDataset, collate_fn
from flows_unexpected_dataset import FlowUnexpectedDataset
from util import get_ground_truth_dict, get_logging_ground_truth_dict, timer

_seed = 42

class Detector:
    """
    A detector is a function that takes a set of flows and returns a set of
    flows that are unusual.
    """

    def __call__(self, specs: FlowSpecs) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        """
        Run the detector on the given set of flows.
        """
        raise NotImplementedError()


class RandomDetector(Detector):
    """
    A detector for finding unusual flows by random choice, optionally weighted
    based on the overall distribution of sink types in the training set.
    """

    def __init__(self, specs: FlowSpecs):
        sinkweights = specs.get_sink_weights()
        self.sinks = list(sinkweights.keys())
        self.weights = list(sinkweights.values()) if specs.options.use_weights else None

    def __call__(self, specs: FlowSpecs) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        unusual_flows = []
        for spec in specs:
            if spec.sink == 'None':
                continue
            expected = random.choices(self.sinks, weights=self.weights)[0]
            if spec.sink != expected:
                unusual_flows.append(spec)

        return unusual_flows, {}


class CountingDetector(Detector):
    """
    A detector for finding unusual flows by a statistical approach: a flow
    spec is considered unusual if this particular combination of parameter name
    and sink only occurs in less than (_rare * 100)% of all flows for this
    parameter.
    """

    def __init__(self, specs: FlowSpecs):
        self.param_count: dict[ParamRepr, int] = defaultdict(int)
        self.param_sink_count: dict[ParamReprSinkPair, int] = defaultdict(int)

        with timer('counting occurrences'):
            for spec in specs:
                param = spec.param_repr()
                self.param_count[param] += 1
                self.param_sink_count[(param, spec.sink)] += 1

    def __call__(self, specs: FlowSpecs) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        with timer('finding unusual flows'):
            unusual_flows = []
            flow_prob_dict = {}
            for (param, sink), instances in specs.ps_map.items():
                key = (param, sink)
                if sink != 'None': 
                    if key in self.param_sink_count:
                        freq = self.param_sink_count[key] / self.param_count[param]
                        if freq > 0 and freq < specs.options.rarity_threshold:
                            unusual_flows.extend(instances)
                        for flow in instances:
                            flow_prob_dict[flow] = freq
                    else:
                        # we have never seen this flow, 
                        # which means this param does not flow into other sinks
                        # so we consider it an usual flow for this sink
                        for flow in instances:
                            flow_prob_dict[flow] = 1
        return unusual_flows, flow_prob_dict


class ClassifierDetector(Detector):
    """
    A detector for finding unusual flows using a random-forest classifier on
    identifier embeddings, optionally including class weights based on the
    overall distribution of sink types.

    The detector checks the prediction probabilities for each parameter-sink
    pair, and flag those where the probability is below _rare as unusual.
    """

    def __init__(self, specs: FlowSpecs, embeddings: Encoder):
        with timer("classifier training"):
            with timer("embedding param_repr"):
                param_embeddings = specs.get_param_embeddings(embeddings)
            sink_embeddings = [spec.sink for spec in specs]
            weights = specs.get_sink_weights() if specs.options.use_weights else None
            self.forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight=weights)
            self.forest.fit(param_embeddings, sink_embeddings)
            self.embeddings = embeddings

    def __call__(self, specs: FlowSpecs) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        with timer("finding unusual flows"):
            unusual_flows = []
            flow_prob_dict = {}
            for (param, sink), instances in specs.ps_map.items():
                if sink != "None":
                    probs = self.forest.predict_proba([param.embed(self.embeddings)])[0]
                    class_idx = self.forest.classes_.tolist().index(sink)  # type: ignore
                    prob = probs[class_idx]
                    if prob > 0 and prob < specs.options.rarity_threshold:
                        unusual_flows.extend(instances)
                    for flow in instances:
                        flow_prob_dict[flow] = prob

        return unusual_flows, flow_prob_dict


class RegressorDetector(Detector):
    """
    A detector for finding unusual flows using a logistic regression model on
    identifier embeddings.

    The detector checks the probability the regressor assigns to actually
    observed sink kinds, and flag those where the probability is below _rare as
    unusual.
    """

    def __init__(self, specs: FlowSpecs, embeddings: Encoder):
        self.sink_types = sorted(set([spec.sink for spec in specs]))
        with timer("regression model training"):
            with timer("embedding param_repr"):
                param_embeddings = specs.get_param_embeddings(embeddings)
            sink_embeddings = [spec.sink for spec in specs]
            weights = specs.get_sink_weights() if specs.options.use_weights else None
            self.model = LogisticRegression(
                solver='saga',
                multi_class='multinomial',
                penalty='l1',
                class_weight=weights
            )
            self.model.fit(param_embeddings, sink_embeddings)
            self.embeddings = embeddings

    def __call__(self, specs: FlowSpecs) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        with timer("finding unusual flows"):
            unusual_flows = []
            flow_prob_dict = {}
            for (param, sink), instances in specs.ps_map.items():
                if sink != "None":
                    probs = self.model.predict_proba([param.embed(self.embeddings)])[0]
                    class_idx = self.model.classes_.tolist().index(sink)
                    prob = probs[class_idx]
                    if prob > 0 and prob < specs.options.rarity_threshold:
                        unusual_flows.extend(instances)
                    for flow in instances:
                        flow_prob_dict[flow] = prob

        return unusual_flows, flow_prob_dict


class NeuralClassifierDetector(Detector):
    """
    A detector for finding unusual flows using a feedforward neural network on
    identifier embeddings, optionally including class weights (used in the loss function)
    based on the overall distribution of sink types.

    The detector checks the prediction probabilities (via softmax layer) for each parameter-sink
    pair, and flag those where the probability is below _rare as unusual.
    """

    # , embeddings: Encoder):
    def __init__(self, specs: FlowSpecs, use_weighted_sampler: bool = False):
        with timer("NN classifier training"):
            _seed_torch(_seed)
            g = torch.Generator()
            g.manual_seed(_seed)
            with timer("producing FlowDataset"):
                sink_list = [spec.sink for spec in specs]
                # store output label (in case test data does not have all the target labels)
                output_label = sorted(set(sink_list))
                tokenizer = get_tokenizer('basic_english')
                vocab = _build_vocab(tokenizer, specs)
                text_pipeline = lambda x: vocab(tokenizer(x))
                dataset = FlowDataset(
                    [spec for spec in specs], output_label, text_pipeline)
                train_data, val_data = _balance_val_split(dataset, 0.1, _seed)
                # For weighted sampler: calculate weight for each sink
                if use_weighted_sampler:
                    targets = [d[1] for d in train_data]
                    counts = np.bincount(targets)
                    labels_weights = 1. / counts
                    weights = labels_weights[targets]
                    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
                    train_loader = DataLoader(
                        train_data, batch_size=256, collate_fn=collate_fn, sampler=sampler, worker_init_fn=_seed_worker, generator=g)
                else:
                    train_loader = DataLoader(
                        train_data, batch_size=256, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=g)
                val_loader = DataLoader(
                    val_data, batch_size=256, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=g)

            # init model
            if specs.options.use_weights:
                class_weights = class_weight.compute_class_weight(
                    'balanced', classes=np.unique(output_label), y=sink_list)
                classifier = ClassifierNet(include_fn=specs.options.include_function_name, include_doc=specs.options.include_param_doc,
                                           output_label=output_label, vocab=vocab, class_weights=class_weights)
            else:
                classifier = ClassifierNet(include_fn=specs.options.include_function_name, include_doc=specs.options.include_param_doc,
                                           output_label=output_label, vocab=vocab)

            # Train
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    mode="min",
                    patience=8,  # i.e. stop if min val_loss does not decrease in 2 epochs
                ),
                ModelCheckpoint(
                    monitor='val_loss',
                    mode="min",
                    filename='model-{epoch:02d}-{val_loss:.3f}'
                ),
            ]
            trainer = pl.Trainer(
                accelerator='gpu', max_epochs=10, val_check_interval=0.25, callbacks=callbacks)
            trainer.fit(model=classifier, train_dataloaders=train_loader,
                        val_dataloaders=val_loader)

            # Test
            # Note: the best model checkpoint from the previous trainer.fit call
            # will be loaded if a checkpoint callback is configured.
            trainer.test(dataloaders=val_loader)

    def __call__(self, specs: FlowSpecs, model_path: Path) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        with timer("finding unusual flows"):
            with timer("loading model"):
                # Load the model from path
                # Note: any arguments specified through **kwargs will override args stored in "hyper_parameters".
                # e.g. classifier = ClassifierNet.load_from_checkpoint(model_path, include_fn=True)
                classifier = ClassifierNet.load_from_checkpoint(str(model_path))
            with timer("producing FlowDataset"):
                # only output spec that flows into a sink
                specs_filtered = [
                    spec for spec in specs if spec.sink != 'None']
                # load output label
                tokenizer = get_tokenizer('basic_english')
                text_pipeline = lambda x: classifier.vocab(tokenizer(x))
                # text_pipeline = lambda x: classifier._hparams['vocab'](tokenizer(x))
                dataset = FlowDataset(specs_filtered, classifier.output_label, text_pipeline)
                g = torch.Generator()
                g.manual_seed(_seed)
                test_dataloader = DataLoader(
                    dataset, batch_size=256, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=g)
            unusual_flows = set()
            flow_prob_dict = {}

            trainer = pl.Trainer(accelerator='gpu')
            probs = trainer.predict(
                model=classifier, dataloaders=test_dataloader)
            # probs is a list of tensor,
            # probs[0].shape == (N,5), where N is len(specs_filtered), 5 is the number
            # of sink classes
            probs_flattened = [p for elem in probs for p in elem]

            for i in range(len(dataset)):
                input, sink = dataset[i]
                if probs_flattened[i][sink] >= 0 and probs_flattened[i][sink] < specs.options.rarity_threshold:
                    unusual_flows.add(input[0])
                flow_prob_dict[input[0]] = probs_flattened[i][sink].numpy()

        return unusual_flows, flow_prob_dict

    def fine_tune(
        self,
        specs: FlowSpecs,
        model_path: Optional[Path],
        sink: SinkKind,
        dataset_size_experiment: bool = False,
        include_random_flows: bool = False,
        include_secbenchjs: bool = False,
    ) -> tuple[Iterable[FlowSpec], dict[FlowSpec, np.ndarray]]:
        with timer("Fine-tuning binary classifier"):
            _seed_torch(_seed)
            with timer("loading model"):
                if model_path:
                    # Load the model from path
                    classifier = ClassifierNet.load_from_checkpoint(str(model_path))
                else:
                    # Train classifier without a pre-trained model
                    tokenizer = get_tokenizer('basic_english')
                    vocab = _build_vocab(tokenizer, specs)
                    classifier = ClassifierNet(include_fn=True, include_doc=True,
                                               output_label=['Expected', 'Unexpected'], vocab=vocab)
                # Replace output layer of the model
                classifier.output_label = ['Expected', 'Unexpected']
                layer_to_remove = list(classifier.classifier.children())[:-1] # i.e. the last layer (250,5)
                feedforward = torch.nn.Sequential(*layer_to_remove)
                # add a new layer with 2 output classes
                classifier.classifier = torch.nn.Sequential(feedforward, torch.nn.Linear(250,len(classifier.output_label)))
                # Reset CE loss, in case the model uses weighted loss
                classifier.loss = torch.nn.CrossEntropyLoss()
                # Reset F1 score
                classifier.f1 = F1Score(average='micro')
            with timer("producing FlowUnexpectedDataset"):
                # Read flows from ground-truth.csv (only 1 sink type)
                if sink == 'logging':
                    ground_truth_dict = get_logging_ground_truth_dict()
                else:
                    ground_truth_dict = get_ground_truth_dict()
                tokenizer = get_tokenizer('basic_english')
                text_pipeline = lambda x: classifier.vocab(tokenizer(x))

                # Produce a unexpected flows dataset (label is expected),
                # one dataset per same sink type
                ground_truth_of_sink = {spec: exp for spec, exp in ground_truth_dict.items() if spec.sink == sink}
                dataset = FlowUnexpectedDataset(ground_truth_of_sink, text_pipeline)                
                extra_eval_datasets = {}
                if include_random_flows:
                    random_ground_truth = get_ground_truth_dict(ground_truth_path='data/ground-truth-full-flows.csv')
                    random_ground_truth_of_sink = {spec: exp for spec, exp in random_ground_truth.items() if spec.sink == sink}
                    random_dataset = FlowUnexpectedDataset(random_ground_truth_of_sink, text_pipeline)
                    extra_eval_datasets['Random set'] = random_dataset
                if include_secbenchjs:
                    secbench_ground_truth = get_ground_truth_dict(ground_truth_path='data/SecBench.js.csv')
                    secbench_ground_truth_of_sink = {spec: exp for spec, exp in secbench_ground_truth.items() if spec.sink == sink}
                    secbench_dataset = FlowUnexpectedDataset(secbench_ground_truth_of_sink, text_pipeline)
                    extra_eval_datasets['SecBench.js'] = secbench_dataset
            with timer("running k-flod cross-validation"):
                # Use k-fold cross-val for the small manually labelled dataset
                k = 5
                kfold = KFold(n_splits=k, shuffle=True, random_state=_seed)
                state_dict = classifier.state_dict()
                if dataset_size_experiment:
                    # multiply these number to get 70,60,...,10% of the original dataset (which is 80% by default)
                    dataset_ratios = [i/(1-1/k) for i in [.7, .6, .5, .4, .3, .2, .1]]
                    for r in dataset_ratios:
                        _run_kfold(kfold, dataset, classifier, state_dict, dataset_ratio=r)
                    # return empty result
                    return set(), {}
                else:
                    return _run_kfold(kfold, dataset, classifier, state_dict, extra_eval_datasets=extra_eval_datasets)

def _seed_torch(seed=42):
    """
    Seed everything for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # ref: https://pytorch.org/docs/1.11/generated/torch.nn.LSTM.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # setting torch.backends.cudnn.enabled to false will slow down the training,
    # but doc_lstm network will not be deterministic if it is set to true
    torch.backends.cudnn.enabled = False
    pl.seed_everything(seed, workers=True)
    # torch.use_deterministic_algorithms(mode=True)
    torch.set_deterministic_debug_mode('error')
    torch.set_printoptions(precision=10)

def _seed_worker(worker_id):
    """
    DataLoader will reseed workers following Randomness in multi-process data loading algorithm. 
    Use worker_init_fn() and generator to preserve reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _run_kfold(kfold: KFold,
               dataset: FlowUnexpectedDataset,
               classifier: ClassifierNet,
               state_dict,
               dataset_ratio: Optional[float] = None,
               extra_eval_datasets: dict[str, FlowUnexpectedDataset] = {}):
    unusual_flows = set()
    flow_prob_dict = {}
    f1_dict = defaultdict(list)
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        with timer(f"k-fold {fold}: training"):
            # for each fold, re-initialize the classifier
            classifier.load_state_dict(state_dict)
            if dataset_ratio:
                dataset_size = int(dataset_ratio * len(train_index))
                train_index = np.random.choice(train_index, size=dataset_size, replace=False)
            else:
                dataset_size = len(train_index)
            # Split into train/val set
            train_data = Subset(dataset, indices=train_index)
            test_data = Subset(dataset, indices=test_index)
            # Build dataloader
            g = torch.Generator()
            g.manual_seed(_seed)
            train_loader = DataLoader(
                train_data, batch_size=32, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=g)
            test_loader = DataLoader(
                test_data, batch_size=32, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=g)

            callbacks = [
                EarlyStopping(
                    monitor='train_loss_epoch',
                    mode="min",
                    patience=50,  # i.e. stop if min train_loss_epoch does not decrease in 50 epochs
                ),
                ModelCheckpoint(
                    monitor='train_loss_epoch',
                    mode="min",
                    filename='model-{epoch:02d}-{train_loss:.3f}',
                    save_on_train_epoch_end=True,
                ),
            ]

            trainer = pl.Trainer(
                accelerator='gpu', max_epochs=500, callbacks=callbacks,
                # progress_bar_refresh_rate=0,
                # val_check_interval=0.25, # don't use this for k-fold cross-val
                )
            trainer.fit(model=classifier, train_dataloaders=train_loader)
            # trainer.fit(model=classifier, train_dataloaders=train_loader,
            #             val_dataloaders=val_loader)
            trainer.test(dataloaders=test_loader)
        with timer(f"k-fold {fold}: evaluation"):
            # Evaluation:
            probs = trainer.predict(
                model=classifier, dataloaders=test_loader)
            # probs is a list of tensor,
            # probs[0].shape == (N,2), where N is len(specs_filtered), 2 is the [exp, unexp]        
            f1 = _compute_stats(test_data, probs, 'Balanced set')
            f1_dict['Balanced set'].append(f1)
            if extra_eval_datasets:
                for dataset_name, extra_eval_dataset in extra_eval_datasets.items():
                    # remove the flows in extra_eval_dataset that are also in the training set
                    training_flows = [t for (t,_), _ in train_data]
                    dataset_indices = [i for i,((f,_),_) in enumerate(extra_eval_dataset) if f not in training_flows]
                    extra_eval_dataset_filtered = Subset(extra_eval_dataset, indices=dataset_indices)
                    extra_eval_loader = DataLoader(extra_eval_dataset_filtered, batch_size=32, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=g)
                    # evaluate on the extra eval dataset
                    if len(extra_eval_loader) > 0:
                        probs = trainer.predict(model=classifier, dataloaders=extra_eval_loader)
                        f1 = _compute_stats(extra_eval_dataset_filtered, probs, dataset_name)
                        f1_dict[dataset_name].append(f1)
    for dataset_name in f1_dict.keys():
        _print_avg_stats(f1_dict[dataset_name], dataset_name)
    if dataset_ratio:
        print(f"Training dataset size: {dataset_size} ({dataset_ratio*(1-1/kfold.get_n_splits())})", file=sys.stderr)
    else:
        print(f"Training dataset size: {dataset_size}", file=sys.stderr)
    return unusual_flows, flow_prob_dict

def _build_vocab(tokenizer, specs: FlowSpecs) -> Vocab:
    def yield_tokens(specs: FlowSpecs):
        for spec in specs:
            yield tokenizer(spec.param.param_doc)
    vocab = build_vocab_from_iterator(yield_tokens(specs), specials=["<unk>"], min_freq=3)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def _balance_val_split(dataset: FlowDataset, val_split=0.1,
                       random_state: int = 42) -> tuple[Subset, Subset]:
    """
    Split the dataset into train and validation sets.
    This function will keep the distribution of the target label the same.

    e.g. if 95% of the dataset are `None` flows,
    then 95% of the train and validation sets are `None` flows respectively.
    """
    targets = np.array(dataset.target)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets,
        random_state=random_state
    )
    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset

def _compute_stats(test_data: Dataset, probs: list[torch.Tensor], dataset_name: str):
    probs_flattened = [p for elem in probs for p in elem]
    tps = 0
    fps = 0
    tns = 0
    fns = 0
    y_true = []
    y_pred = []
    for i in range(len(test_data)):
        input, unexpected = test_data[i]
        y_true += [1] if unexpected else [0]
        if probs_flattened[i][0] < probs_flattened[i][1]:
            # classifier says unexpected
            # unusual_flows.add(input[0])
            y_pred += [1]
            if unexpected:
                tps += 1
            else:
                fps += 1
        else:
            # classifier says expected
            y_pred += [0]
            if unexpected:
                fns += 1
            else:
                tns += 1
        # flow_prob_dict[input[0]] = probs_flattened[i][1].numpy()
    print('Dataset:', dataset_name, file=sys.stderr)
    print(classification_report(y_true, y_pred, target_names=['expected', 'unexpected']), file=sys.stderr)
    precision = tps / (tps + fps) if tps + fps > 0 else 0
    recall = tps / (tps + fns) if tps + fns > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0
    print(
        f"Precision/Recall/F1: {precision:.3f}/{recall:.3f}/{f1:.3f}", file=sys.stderr)
    return f1

def _print_avg_stats(f1_list: list[float], dataset_name: str) -> float:
    avg_f1 = sum(f1_list) / len(f1_list)
    std = np.std(f1_list)
    print('Dataset:', dataset_name, file=sys.stderr)
    print(f"F1-scores in folds: {f1_list}" , file=sys.stderr)
    print(f"Average F-1 score cross validation: {avg_f1:.3f}" , file=sys.stderr)
    print(f"Standard deviation: {std:.3f}" , file=sys.stderr)
