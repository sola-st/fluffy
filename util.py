import csv
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Optional
from config import Options

from flows import FlowSpec, FlowSpecs, skip


@contextmanager
def timer(task: str) -> Iterator[float]:
    logging.info(f'Starting {task}')
    start = perf_counter()
    try:
        yield start
    finally:
        logging.info(
            f'Finished {task} in {perf_counter() - start:.2f} seconds')

def get_ground_truth_dict(specs: Optional[FlowSpecs] = None, ground_truth_path: str = 'data/ground-truth.csv') -> dict[FlowSpec, bool]:
    """
    Returns a dictionary reading ground truth from `ground_truth_path`.
    A `FlowSpec` is mapped to True if it is marked as unexpected in `ground_truth_path`, False otherwise.
    If `specs` is supplied, removes the ground truth that are not in `specs`.
    """
    # set a dummy `options` with no effect if `specs` is not given
    options = Options({}) if specs is None else specs.options
    ground_truth: dict[FlowSpec, bool] = {}
    with open(ground_truth_path, 'r') as f:
    # with open('data/ground-truth-full-flows.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        i = headers.index('Unusual')
        if i == -1:
            raise ValueError('No "Unusual" column in ground-truth.csv')
        for row in reader:
            if i >= len(row):
                raise ValueError(f'Row too short: {row}')
            flow_spec = FlowSpec.from_row(row[0:i], options, has_metadata=False)
            if flow_spec.sink == 'None' or skip(flow_spec.param.parameter) or (specs is not None and flow_spec not in set(specs.specs)):
                # skip:
                # - specs flowing into None
                # - specs with very short names (we do the same for training set)
                # - specs that are not in the test set we read from
                continue
            if row[i] == 'yes':
                ground_truth[flow_spec] = True
            elif row[i] == 'no':
                ground_truth[flow_spec] = False
    return ground_truth

def get_logging_ground_truth_dict(ground_truth_path: str = 'data/logging_flows_ground_truth.csv') -> dict[FlowSpec, bool]:
    """
    Returns a dictionary reading ground truth from `ground_truth_path`.
    Same as `get_ground_truth_dict()` but for logging flows.
    A `FlowSpec` is mapped to True if it is marked as 'TP'/'FN' in `ground_truth_path`, False otherwise.
    """
    # set a dummy `options` with no effect
    options = Options({}) 
    ground_truth: dict[FlowSpec, bool] = {}
    with open(ground_truth_path, 'r') as f:
    # with open('data/ground-truth-full-flows.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        i = headers.index('label')
        if i == -1:
            raise ValueError('No "label" column in ground-truth.csv')
        for row in reader:
            if i >= len(row):
                raise ValueError(f'Row too short: {row}')
            flow_spec = FlowSpec.from_logging_row(row[0:i], options, has_metadata = False, with_none_sink = False)
            if skip(flow_spec.param.parameter):
                # skip:
                # - specs with very short names (we do the same for training set)
                continue
            if row[i] == 'TP' or row[i] == 'FN':
                ground_truth[flow_spec] = True
            elif row[i] == 'TN' or row[i] == 'FP':
                ground_truth[flow_spec] = False
    return ground_truth