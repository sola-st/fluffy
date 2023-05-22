import csv
import re
from collections import defaultdict
from typing import Iterable, Iterator, Literal, Union, cast

import numpy as np
from varclr.models.model import Encoder

import config

SinkKind = Literal['None',
                   'CodeInjection', 'CommandInjection',
                   'ReflectedXss', 'TaintedPath', 'logging']
"A sink kind to which a parameter flows, or None if it does not flow to any known sink."


class ParamSpec:
    """
    A parameter from an API, characterized by the following attributes:
    - parameter: the name of the parameter
    - function: the function to which the parameter belongs
    - package: the package to which the function belongs
    - param_doc: the doc comment of the parameter (`@param` tag)
    """

    def __init__(self: 'ParamSpec', package: str, function: str, parameter: str, param_doc: str):
        self.package = package
        self.function = function
        self.parameter = parameter
        self.param_doc = param_doc

    def to_row(self: 'ParamSpec') -> list[str]:
        "Convert a parameter spec into a CSV row."
        return [self.package, self.function, self.parameter, self.param_doc]

    def __str__(self: 'ParamSpec') -> str:
        return ','.join(self.to_row())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParamSpec):
            return False
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class FlowSpec:
    """
    A flow from an API parameter to a sink or a non-sink.
    """
    param: ParamSpec
    sink: SinkKind
    options: config.Options

    def __init__(self: 'FlowSpec', param: ParamSpec, sink: SinkKind, options: config.Options):
        self.param = param
        self.sink = sink
        self.options = options

    def param_repr(self: 'FlowSpec') -> 'ParamRepr':
        return ParamRepr(self.param, self.options)

    @staticmethod
    def from_row(row: list[str], options: config.Options, has_metadata: bool = False) -> 'FlowSpec':
        "Read a flow spec from a CSV row."
        if has_metadata:
            if len(row) != 9:
                raise ValueError(f'Expected 9 columns, got {len(row)}: {row}')
            _, package, _, function, parameter, sink, param_doc, _, _ = row
        else:
            if len(row) == 5:
                package, function, parameter, param_doc, sink = row
            else:
                raise ValueError(f'Expected 5 columns, got {len(row)}: {row}')
        sink_filter = lambda sink, single_sink: 'None' if single_sink and sink != single_sink else sink
        return FlowSpec(ParamSpec(package, function, parameter, param_doc), cast(SinkKind, sink_filter(sink, options.single_sink)), options)

    @staticmethod
    def from_logging_row(row: list[str],
                         options: config.Options,
                         has_metadata: bool = False,
                         with_none_sink: bool = True) -> 'FlowSpec':
        """
        Read a logging flow spec from a CSV row. All flows are considered to have a `logging` sink.
        Unless `with_none_sink` is `True` (by default) and `has_metadata` is `True`, in that case,
        a flow is considered to have a `None` sink if it is flagged as not problematic by the CodeQL query.
        """
        if has_metadata:
            if len(row) != 8:
                raise ValueError(f'Expected 8 columns, got {len(row)}: {row}')
            if with_none_sink:
                _, _, _, parameter, _, _, sink, flag = row
                if flag == 'false':
                    sink = 'None'
            else:
                _, _, _, parameter, _, _, sink, _ = row
        else:
            if len(row) == 2:
                parameter, sink = row
            else:
                raise ValueError(f'Expected 2 columns, got {len(row)}: {row}')
        return FlowSpec(ParamSpec('', '', parameter, ''), cast(SinkKind, sink), options)

    def to_row(self: 'FlowSpec') -> list[str]:
        "Convert a flow spec into a CSV row."
        return self.param.to_row() + [self.sink]

    def __str__(self: 'FlowSpec') -> str:
        return ','.join(self.to_row())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowSpec):
            return False
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class ParamRepr:
    """
    A representation of an API parameter for analysis purposes, including the
    parameter name and optionally the name of the function to which it belongs.
    """

    def __init__(self: 'ParamRepr', param: ParamSpec, options: config.Options):
        self.parameter = param.parameter
        self.function = param.function if options.include_function_name else None
        self.options = options

    def embed(self: 'ParamRepr', embeddings: Encoder):
        if self.options.include_function_name:
            return np.concatenate([
                embeddings.encode(self.function).cpu().numpy(),  # type: ignore
                embeddings.encode(self.parameter).cpu().numpy()  # type: ignore
            ], axis=1)[0]
        else:
            return embeddings.encode(self.parameter).cpu().numpy()[0]  # type: ignore

    def __str__(self: 'ParamRepr') -> str:
        if self.options.include_function_name:
            return f'{self.parameter}@{self.function}'
        else:
            return self.parameter

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParamRepr):
            return False
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


ParamReprSinkPair = tuple[ParamRepr, SinkKind]
"A parameter representation and a sink kind it flows to."


class FlowSpecs(Iterable[FlowSpec]):
    """
    A collection of flow specs, which can be retrieved as a plain list,
    as a map from parameter-function-sink pairs to lists of flow specs, or
    as a map from parameter-sink pairs to lists of flow specs.
    """

    def __init__(self: 'FlowSpecs', specs: list[FlowSpec], options: config.Options):
        self.specs = specs
        self.options = options
        self.ps_map: dict[ParamReprSinkPair,
                          list[FlowSpec]] = defaultdict(lambda: [])
        for spec in specs:
            self.ps_map[(spec.param_repr(), cast(SinkKind, spec.sink))].append(spec)

    def get_sink_weights(self: 'FlowSpecs'):
        """
        Count how often each sink kind appears in this collection of flow specs.
        """
        sinkweights = {}
        for spec in self:
            sink = spec.sink
            if sink not in sinkweights:
                sinkweights[sink] = 0
            sinkweights[sink] += 1
        return sinkweights

    def get_param_embeddings(self: 'FlowSpecs', embeddings: Encoder):
        """
        Compute the parameter embeddings using `VarCLR`.
        """
        # We separate VarCLR to use batch encoding since calling encode() for
        # each spec is time-consuming (~0.02 second for each spec)
        param_embeddings = []
        batch_size = 1024
        for i in range(0, len(self), batch_size):
            batch = self.specs[i:i + batch_size]
            if self.options.include_function_name:
                param_embeddings.extend(
                    np.concatenate([
                        embeddings.encode(  # type: ignore
                            [spec.param.function for spec in batch]).cpu().numpy(),
                        embeddings.encode(  # type: ignore
                            [spec.param.parameter for spec in batch]).cpu().numpy()
                    ], axis=1)
                )
            else:
                param_embeddings.extend(embeddings.encode(  # type: ignore
                    [spec.param.parameter for spec in batch]).cpu().numpy())
        return param_embeddings

    def __iter__(self: 'FlowSpecs') -> Iterator[FlowSpec]:
        return iter(self.specs)

    def __len__(self: 'FlowSpecs'):
        return len(self.specs)


def tokenize(name):
    """
    Split the name at underscores, numbers and lower-to-upper case transitions.
    """
    return re.split(r"_|\d+|(?<=[a-z])(?=[A-Z])", name)


def is_generic(word):
    "Check if the given word is very generic."
    return word in ["arg", "result", "temp", "tmp", "sample"]


def is_short(word):
    "Check if the given word is very short."
    return len(word) <= 1


def skip(name):
    """
    Check if the given name should be skipped because it contains a very short token.
    """
    return is_short(name)


def read_flows(file, options: config.Options, has_metadata: bool = True, is_logging: bool = False) -> FlowSpecs:
    """
    Read flow specs from the given file, which is expected to be in the
    format produced by the flows2csv.py script, except for flows with a logging sink.
    """
    with open(file, "r") as f:
        if is_logging:
            specs = [FlowSpec.from_logging_row(row, options, has_metadata=has_metadata, with_none_sink=True)
                 for row in csv.reader(f)]
        else:
            specs = [FlowSpec.from_row(row, options, has_metadata=has_metadata)
                 for row in csv.reader(f)]
    return FlowSpecs([spec for spec in specs if not skip(spec.param.parameter)], options)


def write_flows(flows: Iterable[FlowSpec], file):
    """
    Write the given list of flow specs to the given file in CSV format.
    """
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(sorted([spec.to_row() for spec in flows]))
