import collections
import torch
from torch.utils.data import Dataset
from flows import FlowSpec, SinkKind

class FlowDataset(Dataset):
    """
    Represent the flow specs as a torch dataset, 
    which can be used by `torch.utils.data.DataLoader`.
    """
    def __init__(self, specs: list[FlowSpec], sink_types: set[SinkKind], text_pipeline):
        self.sink_types = sink_types
        flow_specs = specs
        # processed_doc_comment should be a list of int, where each int is an index of the word in the vocab
        processed_doc_comment = [text_pipeline(spec.param.param_doc) if spec.param.param_doc else [0] for spec in specs]
        self.input = tuple(zip(flow_specs, processed_doc_comment))
        # `target` is a list where each entry represents the sink of the flow.
        # Values in `target` can be {0, 1, 2, 3, 4}, where the number represents the sink class.
        self.target = [self.sink_types.index(spec.sink) for spec in specs]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx) -> tuple[tuple[FlowSpec, list[int]], int]:
        """
        Returns a sample from the dataset. Must be a pair of (input, label).
        """
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # label = torch.tensor(self.target_encodings["input_ids"][idx], dtype=torch.long)
        return self.input[idx], self.target[idx]

def collate_fn(batch):
    """
    For PyTorch DataLoader: merges a list of samples to form a mini-batch of Tensor(s).  
    Used when using batched loading from a map-style dataset.
    
    The checks come from `default_collate` from `torch.utils.data._utils.collate`, but allow using the `FlowSpec` type.
    """

    collate_err_msg_format = (
    "collate_fn: batch must be tuple, int or FlowSpec; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, int): # handle the processed_doc_comment, label (sink)
        return torch.tensor(batch)
    elif isinstance(elem, FlowSpec): # handle the flow_specs
        return batch
    elif isinstance(elem, collections.abc.Sequence): # `FlowDataset` returns a tuple
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate_fn(samples) for samples in transposed]  # Backwards compatibility.
        elif isinstance(elem, list): # param doc comment list
            # add padding
            max_doc_len = max(len(doc) for doc in batch)
            batch = [doc + [0]*(max_doc_len - len(doc)) for doc in batch]
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            return torch.tensor(batch)

    raise TypeError(collate_err_msg_format.format(elem_type))

