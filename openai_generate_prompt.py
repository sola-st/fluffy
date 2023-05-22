import random
from textwrap import dedent
from typing import Literal, Optional
from flows import FlowSpec

PromptType = Literal['DirectPrediction', 'SimilarityCheck']

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
    },
    'logging': {
        'normal': ['authkey', 'password', 'passcode', 'passphrase'],
        'unusual': []
    }
}

sink_descriptions = {
    'CommandInjection': 'arbitrary command line execution',
    'CodeInjection': 'arbitrary code execution',
    'ReflectedXss': 'reflected cross-site scripting',
    'TaintedPath': 'uncontrolled data used in path expression',
    'logging': ''
}

def get_prompt_from_csv(ground_truth, sink, prompt_type: PromptType, query_spec_tuple: Optional[tuple[FlowSpec, bool]] = None, text_file=None):
    ground_truth_list = list(ground_truth.items())
    ground_truth_list.sort(key=lambda x: str(x[0].param_repr()))
    full_str = ''
    if query_spec_tuple:
        current_spec, current_label = query_spec_tuple
        seed = hash(current_spec)
    else:
        seed = 42
    random.Random(seed).shuffle(ground_truth_list)
    examples_count = 10
    # Type 1 (as code: predict by parameter name, function name, and parameter docstring)
    if prompt_type == 'DirectPrediction':
        # f.write(f'Classify the following parameter, function, and optional parameter document to "expected" or "unexpected" for arbitrary path expression.\n')
        for spec, is_unusual in ground_truth_list:
            if examples_count <= 0:
                break
            elif spec.sink == sink and (not query_spec_tuple or spec != current_spec):
                full_str += prompt_text_one(spec, is_unusual, sink)
                examples_count -= 1
        if current_spec:
            full_str += prompt_text_one(current_spec, current_label, sink, show_expected=False)
        if text_file:
            with open(text_file, 'w') as f:
                f.write(full_str)
        return full_str
    # Type 2 (similarity check)
    elif prompt_type == 'SimilarityCheck':
        normal_samples = samples[sink]['normal']
        # surrond normal_samples with double quotes
        normal_samples_str = ', '.join([f'"{s}"' for s in normal_samples])
        full_str += f'Answer "yes" or "no" to whether the following words are similar to {normal_samples_str}.\n\n'
        for spec, is_unusual in ground_truth_list:
            if examples_count <= 0:
                break
            elif spec.sink == sink and (not query_spec_tuple or spec != current_spec):
                full_str += prompt_text_two(spec, is_unusual, normal_samples_str)
                examples_count -= 1
        if current_spec:
            full_str += prompt_text_two(current_spec, current_label, normal_samples_str, show_expected=False)
        if text_file:
            with open(text_file, 'w') as f:
                f.write(full_str)
        return full_str
    else:
        raise ValueError(f'Unknown prompt type {prompt_type}')
    # Type 3 (as text: predict by parameter name, function name, and parameter docstring)
    # with open(text_file, 'w') as f:
    #     ground_truth = get_ground_truth_dict(ground_truth_path=csv_file)
    #     ground_truth_list = list(ground_truth.items())
    #     random.shuffle(ground_truth_list)
    #     count = 0
    #     f.write(f'Classify the following parameter, function, and optional parameter document to "expected" or "unexpected" for arbitrary path expression.\n')
    #     for spec, is_unusual in ground_truth_list:
    #         if spec.sink == sink:
    #             param_doc = spec.param.param_doc.replace("\n"," ")
    #             txt = f'Parameter {spec.param.parameter} of function {spec.param.function} with parameter document "{param_doc}" is:'
    #             if count > 0:
    #                 txt += ' unexpected' if is_unusual else ' expected'
    #                 count -= 1
    #             txt += '\n'
    #             f.write(txt)

def prompt_text_one(spec, is_unusual, sink, show_expected: bool = True):
    description = sink_descriptions[sink]
    param_doc = spec.param.param_doc.replace("\n"," ")
    # Parameter {spec.param.parameter} of function {spec.param.function} with parameter document "{param_doc}" is:
    if sink == 'logging':
        txt = f'''
        function f({spec.param.parameter}) {{
          console.log({spec.param.parameter});
        }}
        // In the above function f, the parameter "{spec.param.parameter}" is being logged, which likely exposes'''
        if show_expected:
            txt += ' sensitive data.' if is_unusual else ' insensitive data.'
            txt += '\n'
    else:
        txt = f'''
        /** 
        * @param {spec.param.parameter} - {param_doc}
        */
        function {spec.param.function}({spec.param.parameter}) {{
        }}
        // In the above function "{spec.param.function}", the parameter "{spec.param.parameter}" flows into {sink} sink ({description}), which is'''
        if show_expected:
            txt += ' unexpected.' if is_unusual else ' expected.'
            txt += '\n'
    txt = dedent(txt)
    return txt

def prompt_text_two(current_spec, is_unusual, normal_samples_str, show_expected: bool = True):
    # param_doc = current_spec.param.param_doc.replace("\n"," ")
    name = current_spec.param.function + ' ' + current_spec.param.parameter
    name = name.strip()
    txt = f'Q: is "{name}" similar to any of {normal_samples_str}?\n'
    txt += 'A:'
    if show_expected:
        txt += ' No.' if is_unusual else ' Yes.'
        txt += '\n'
    return txt
