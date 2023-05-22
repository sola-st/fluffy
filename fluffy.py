#! /usr/bin/env python

import argparse
from collections import OrderedDict, defaultdict
import csv
import logging
from pathlib import Path
import pickle
import sys
from typing import Iterable

from varclr.models.model import Encoder

from config import Options
from flows import FlowSpec, FlowSpecs, read_flows, skip, write_flows
from train_detector import (ClassifierDetector, CountingDetector, Detector, NeuralClassifierDetector,
                            RandomDetector, RegressorDetector)
from util import get_logging_ground_truth_dict, timer, get_ground_truth_dict
from util_plot import get_model_name, plot_PR_curve, plot_roc_curve

def print_stats(specs: FlowSpecs, ground_truth, unusual_flows: Iterable[FlowSpec]):
    tps, fps, tns, fns, unlabelled = defaultdict(int), defaultdict(
        int), defaultdict(int), defaultdict(int), defaultdict(int)
    for spec, is_unusual in ground_truth.items():
        if spec in unusual_flows:
            if is_unusual:
                tps[spec.sink] += 1
            else:
                fps[spec.sink] += 1
        else:
            if is_unusual:
                fns[spec.sink] += 1
            else:
                tns[spec.sink] += 1

    print(f"Rarity threshold: {specs.options.rarity_threshold}", file=sys.stderr)

    def print_f1(tps, fps, fns, tns, unlabelled):
        print(
            f"TP/FP/FN/TN/unlabelled: {tps}/{fps}/{fns}/{tns}/{unlabelled}", file=sys.stderr)
        precision = tps / (tps + fps) if tps + fps > 0 else 0
        recall = tps / (tps + fns) if tps + fns > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if precision + recall > 0 else 0
        print(
            f"Precision/Recall/F1: {precision:.3f}/{recall:.3f}/{f1:.3f}", file=sys.stderr)

    for sink in sorted(set([spec.sink for spec in specs if spec.sink != 'None'])):
        unlabelled[sink] = len([f for f in unusual_flows if f.sink == sink]) - tps[sink] - fps[sink]
        print(f"\nMetrics for {sink}", file=sys.stderr)
        print_f1(tps[sink], fps[sink], fns[sink], tns[sink], unlabelled[sink])

    print("\nOverall metrics:", file=sys.stderr)
    print_f1(
        sum(tps.values()),
        sum(fps.values()),
        sum(fns.values()),
        sum(tns.values()),
        sum(unlabelled.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find unusual flows from parameters to sinks.')

    subparsers = parser.add_subparsers(dest='subcommand')

    train_parser = subparsers.add_parser('train', help='Train a detector for unusual flows.')
    train_parser.add_argument('-o', '--output', type=str,
                              help='File to store trained detector.')
    train_parser.add_argument('-t', '--type', type=str,
                              choices=['classifier', 'counting', 'random', 'regressor', 'neural'],
                              help='Type of detector to train.')
    train_parser.add_argument('--include-function-name',
                              action='store_true',
                              help='Consider function name to be part of the parameter name.')
    train_parser.add_argument('--include-param-doc',
                              action='store_true',
                              help='Consider parameter doc comment to be part of the parameter name.')
    train_parser.add_argument('--rarity-threshold', '-r',
                              type=float,
                              default=Options.default_rarity_threshold,
                              help='The maximum frequency of a parameter-sink\
                              pair to be considered unusual.'                                                                                                                                                                                       )
    train_parser.add_argument('--use-weights', '-w',
                              action='store_true',
                              help='Whether to use weights when training the detector.')
    train_parser.add_argument('--weighted-sampler', '-ws',
                              action='store_true',
                              help='(For neural net detector only) Whether to use weighted sampler when training the detector.')
    train_parser.add_argument('--verbose',
                              action='store_true',
                              help='Print more information.')
    train_parser.add_argument('training_set',
                              help='CSV file with training data.', default='data/flows.csv')
    train_parser.add_argument('--single-sink', type=str,
                        help='If provided, the classifier will only classify this sink and treat other sinks as `None`.')
    train_parser.add_argument('--logging-flow',
                              action='store_true',
                              help='Train the model using dataset of logging flows, which is in a different format.')

    detect_parser = subparsers.add_parser('detect', help='Detect unusual flows.')
    detect_parser.add_argument('-o', '--output', type=str,
                               help='File to store results (unusual flows) in.')
    detect_parser.add_argument('--output-prob', type=str,
                               help='Pickle file to store probabilities of param-sink pairs in.')
    detect_parser.add_argument('--verbose',
                               action='store_true',
                               help='Print more information.')
    detect_parser.add_argument('--rarity-threshold', '-r',
                               type=float,
                               help='The maximum frequency of a parameter-sink\
                              pair to be considered unusual. (Override the rarity-threshold set during training.)'                                                                                                                                                                                                                                                                                                                                                      )
    detect_parser.add_argument('detector',
                               help='Trained detector to use.',
                               type=str)
    detect_parser.add_argument('test_set',
                               help='CSV file with test data.',
                               default='data/flows.csv')
    detect_parser.add_argument('--with-metadata',
                               action='store_true',
                               help='Whether the test set contains metadata.')
    detect_parser.add_argument('--model-path',
                               help='(Required for using `neural` detector) The path of the neural network model checkpoint to load.',
                               type=Path)

    evaulation_parser = subparsers.add_parser('eval', help='Perform the evaluation experiment (for sink prediction and frequency-based approach).')
    evaulation_parser.add_argument('--verbose',
                               action='store_true',
                               help='Print more information.')
    evaulation_parser.add_argument('detector',
                               help='Trained detector to use.',
                               type=str)
    evaulation_parser.add_argument('--model-path',
                               help='(Required for using `neural` detector) The path of the neural network model checkpoint to load.',
                               type=Path)
    evaulation_parser.add_argument('--logging-flow',
                              action='store_true',
                              help='Set to true if the model to evaluate is for logging flows, which is different from model trained with API flows.')

    finetune_parser = subparsers.add_parser('finetune', help='Finetune (and predict with) a neural network pre-trained on sink prediction.')
    finetune_parser.add_argument('detector',
                                 help='Trained detector to fine-tune.',
                                 type=str)
    finetune_parser.add_argument('test_set',
                                 help='CSV file with test data (this is only used to build vocab).',
                                 default='data/flows.csv')
    finetune_parser.add_argument('sink',
                                 choices=['None',
                                          'CodeInjection', 'CommandInjection',
                                          'ReflectedXss', 'TaintedPath', 'logging'],
                                 help='Which sink to fine-tune on.',
                                 type=str)
    finetune_parser.add_argument('-dse', '--dataset-size-experiment',
                                 action='store_true',
                                 help='Run the finetuning with various dataset sizes for experiment.')
    finetune_parser.add_argument('--model-path',
                                 help='The path of the neural network model checkpoint to fine-tune.',
                                 type=Path)
    finetune_parser.add_argument('-o', '--output', type=str,
                                 help='File to store results (unusual flows) in.')
    finetune_parser.add_argument('--output-prob', type=str,
                               help='Pickle file to store probabilities of param-sink pairs in.')
    finetune_parser.add_argument('--verbose',
                                 action='store_true',
                                 help='Print more information.')
    finetune_parser.add_argument('--with-metadata',
                                 action='store_true',
                                 help='Whether the test set contains metadata.')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.subcommand == 'train':
        options = Options(vars(args))

        with timer("reading training set"):
            specs = read_flows(args.training_set, options, is_logging=args.logging_flow)
        print('Dataset size', len(specs), file=sys.stderr)
        print('Dataset if flows with None-sink are excluded', len([1 for s in specs if s.sink != 'None']), file=sys.stderr)

        detector: Detector
        if args.type == 'counting':
            detector = CountingDetector(specs)
        elif args.type == 'random':
            detector = RandomDetector(specs)
        elif args.type == 'neural':
            detector = NeuralClassifierDetector(specs, args.weighted_sampler)
        else:
            with timer('loading identifier embeddings'):
                embeddings = Encoder.from_pretrained("varclr-codebert", save_path="models/")
                # TODO: in order to use GPU for VarCLR, you need to change the
                # tensor to use GPU in the VarCLR library code
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # embeddings.to(device)

            if args.type == 'classifier':
                detector = ClassifierDetector(specs, embeddings)
            elif args.type == 'regressor':
                detector = RegressorDetector(specs, embeddings)
            else:
                raise ValueError(f"Unknown detector type: {args.type}")

        # store pickled detector
        data = {
            'options': options,
            'detector': detector,
        }
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
    elif args.subcommand == 'detect':
        with timer('loading detector'):
            with open(args.detector, 'rb') as f:
                data = pickle.load(f)
            options: Options = data['options']
            detector: Detector = data['detector']
            print('options:', vars(options), file=sys.stderr)
        with timer('reading test set'):
            specs = read_flows(args.test_set, options, has_metadata=args.with_metadata)

        if args.rarity_threshold:
            options.rarity_threshold = args.rarity_threshold

        if isinstance(detector, NeuralClassifierDetector):
            if args.model_path is None:
                parser.error('Using a neural classifier requires --model-path.')
            unusual_flows, flow_prob_dict = detector(specs, args.model_path)
        else:
            unusual_flows, flow_prob_dict = detector(specs)

        with timer('writing results'):
            output = open(args.output, 'w') if args.output else sys.stdout
            try:
                write_flows(unusual_flows, output)
                if args.output_prob:
                    with open(args.output_prob, 'wb') as f:
                        pickle.dump(flow_prob_dict, f)
            finally:
                if output is not sys.stdout:
                    output.close()

        with timer('computing statistics'):
            print_stats(specs, get_ground_truth_dict(), unusual_flows)
    elif args.subcommand == 'eval':
        with timer('loading detector'):
            with open(args.detector, 'rb') as f:
                data = pickle.load(f)
            options: Options = data['options']
            detector: Detector = data['detector']
            print('options:', vars(options), file=sys.stderr)
        with timer('reading evaluation dataset'):
            eval_datasets = OrderedDict()
            if args.logging_flow:
                # - logging: output prob. -> f1/roc
                eval_datasets['logging'] = get_logging_ground_truth_dict()
                # all flows in eval_datasets['logging'] have 'logging' as sink
            else:
                # - whole: output prob. -> f1/roc
                eval_datasets['balanced set'] = get_ground_truth_dict()

                # - random: output prob. -> f1/roc
                eval_datasets['random set'] = get_ground_truth_dict(ground_truth_path='data/ground-truth-full-flows.csv')

                # - secbench: use best threshold from `whole` -> f1
                eval_datasets['SecBench.js'] = get_ground_truth_dict(ground_truth_path='data/SecBench.js.csv')

        # Evaluate and plot PR curve and ROC curve for each dataset
        for dataset_name, ground_truth_dict in eval_datasets.items():
            with timer(f'evaluating on {dataset_name}'):
                specs = FlowSpecs([spec for spec in ground_truth_dict.keys() if not skip(spec.param.parameter)], options)
                if isinstance(detector, NeuralClassifierDetector):
                    if args.model_path is None:
                        parser.error('Using a neural classifier requires --model-path.')
                    _, flow_prob_dict = detector(specs, args.model_path)
                else:
                    _, flow_prob_dict = detector(specs)
                model_name = get_model_name(detector)
                threshold = plot_PR_curve(ground_truth_dict, flow_prob_dict, dataset_name, model_name, args.logging_flow)
                plot_roc_curve(ground_truth_dict, flow_prob_dict, dataset_name, model_name, args.logging_flow)

                # use the optimal threshold from whole set to evaluate on SecBench.js
                # 'whole set' evaluation will run first as this is the first dataset in the OrderedDict
                if dataset_name == 'balanced set':
                    best_threshold = threshold
                elif dataset_name == 'SecBench.js':
                    # unusual flows are those with prob. < threshold, different sinks have different thresholds
                    unusual_flows = [spec for spec in specs if flow_prob_dict[spec] < best_threshold[spec.sink]]
                    print_stats(specs, ground_truth_dict, unusual_flows)

    elif args.subcommand == 'finetune':
        with timer('loading detector'):
            with open(args.detector, 'rb') as f:
                data = pickle.load(f)
            options: Options = data['options']
            detector: Detector = data['detector']

        with timer('reading test set'):
            specs = read_flows(args.test_set, options, has_metadata=args.with_metadata)

        if not isinstance(detector, NeuralClassifierDetector):
            parser.error('You can only fine-tune a neural classifier.')
        else:
            unusual_flows, flow_prob_dict = detector.fine_tune(
                specs,
                args.model_path,
                args.sink,
                dataset_size_experiment=args.dataset_size_experiment,
                include_random_flows=True,
                include_secbenchjs=True)

        with timer('writing results'):
            output = open(args.output, 'w') if args.output else sys.stdout
            try:
                write_flows(unusual_flows, output)
                if args.output_prob:
                    with open(args.output_prob, 'wb') as f:
                        pickle.dump(flow_prob_dict, f)
            finally:
                if output is not sys.stdout:
                    output.close()
