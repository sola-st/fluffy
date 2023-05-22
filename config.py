from typing import Literal, Optional

class Options:
    default_rarity_threshold = 0.007

    include_function_name: bool
    "Whether to include the function name in the representation of a parameter."

    include_param_doc: bool
    "Whether to include the document (`@param` tag) in the representation of a parameter."

    single_sink: Optional[str] = None
    "If provided, the classifier will only classify this sink and treat other sinks as `None`."

    rarity_threshold: float
    "Threshold below which a parameter-sink pair is considered unusual."

    use_weights: bool
    "Whether to use weights when training the detector."

    def __init__(self, opts):
        self.include_function_name = opts.get('include_function_name', False)
        self.include_param_doc = opts.get('include_param_doc', False)
        self.rarity_threshold = opts.get('rarity_threshold', self.default_rarity_threshold)
        self.use_weights = opts.get('use_weights', False)
        self.single_sink = opts.get('single_sink', self.single_sink)
