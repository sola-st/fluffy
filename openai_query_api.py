import argparse
from collections import defaultdict
import logging
import openai
import sys
import time

from openai_generate_prompt import get_prompt_from_csv
from util import get_ground_truth_dict, get_logging_ground_truth_dict

# Ref: https://beta.openai.com/docs/guides/code/best-practices

csv_file = 'data/ground-truth-full-flows.csv'
ground_truth_random = get_ground_truth_dict(ground_truth_path=csv_file)
csv_file = 'data/ground-truth.csv'
ground_truth_whole = get_ground_truth_dict(ground_truth_path=csv_file)
csv_file = 'data/logging_flows_ground_truth.csv'
logging_ground_truth = get_logging_ground_truth_dict(ground_truth_path=csv_file)
csv_file = 'data/SecBench.js.csv'
secbench_ground_truth = get_ground_truth_dict(ground_truth_path=csv_file)
all_ground_truth = {}
all_ground_truth.update(ground_truth_whole) # this is superset of ground_truth_random
all_ground_truth.update(logging_ground_truth)
all_ground_truth.update(secbench_ground_truth) # this does not have overlap with other ground truth dict

def classify(all_token_logprobs, expected_label, unexpected_label):
    for token_logprobs in all_token_logprobs:
        tokens_sorted = sorted(token_logprobs, key = lambda ele: token_logprobs[ele], reverse=True)
        for token in tokens_sorted:
            token = token.strip()
            if expected_label == token:
                return False
            elif unexpected_label == token:
                return True
    logging.error('No token matches our label for %s', all_token_logprobs)

def update_result(result, pred_unusual, is_unusual):
    if not pred_unusual and is_unusual:
        result['fn'] += 1
    elif not pred_unusual and not is_unusual:
        result['tn'] += 1
    elif pred_unusual and is_unusual:
        result['tp'] += 1
    elif pred_unusual and not is_unusual:
        result['fp'] += 1

def compute_metrics(sink, result):
    logging.info('Final result for sink %s: %s', sink, result)
    tp = result['tp']
    fp = result['fp']
    tn = result['tn']
    fn = result['fn']
    print('Final result for sink', sink)
    print('Number of flows classified:', sum(result.values()))
    print('tp:', tp)
    print('fp:', fp)
    print('tn:', tn)
    print('fn:', fn)
    # calculate f1 score
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)
    except ZeroDivisionError:
        print('ZeroDivisionError when calculating f1 score')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write queries to OpenAI Codex')
    parser.add_argument('openai_api_key', type=str, help='The OpenAI API key')
    parser.add_argument('--all_flows_examples', help='Use all flows as examples instead of only using random flows', action='store_true')
    parser.add_argument('--sink', type=str, default='all', help='The sink to generate prompts for, if not specified, generate prompts for all sinks')
    parser.add_argument('--prompt_type', type=str, default='DirectPrediction', help='The type of prompt to generate: "DirectPrediction" or "SimilarityCheck"')
    parser.add_argument('--rate_limit', help='Make `rate_limit_per_min` API requests per minute (For me, the rate limit is 60 requests per minute)', action='store_true')
    parser.add_argument('--rate_limit_per_min', type=int, default=60, help='API requests per minute (only useful if `rate_limit` is set')
    parser.add_argument('--GPT3', help='Use GPT-3 instead of Codex', action='store_true')
    parser.add_argument('--debug', help="Run this script with DEBUG logging level", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(filename='openai_query_api.log',
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.DEBUG,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename='openai_query_api.log',
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
    openai.api_key = args.openai_api_key
    prompt_type = args.prompt_type
    ground_truth_as_examples = ground_truth_whole if args.all_flows_examples else ground_truth_random
    if args.GPT3:
        model="text-ada-001" # use the cheapest and fastest model for testing api
    else:
        model="code-davinci-002" # only available for private beta

    rate_limit_per_min = args.rate_limit_per_min
    req_count = rate_limit_per_min
    if args.sink == 'all':
        sinks_to_test = [
            'CodeInjection', 'CommandInjection', 'ReflectedXss', 'TaintedPath',
            'logging'
        ]
    else:
        sinks_to_test = [args.sink]

    for sink in sinks_to_test:
        result_dict = defaultdict(lambda: defaultdict(int))
        # generate prompt for each spec
        for spec, is_unusual in all_ground_truth.items():
            if spec.sink == sink:
                if args.rate_limit:
                    if req_count <= 0:
                        # wait for 1 min and refresh req_count
                        time.sleep(60)
                        req_count = rate_limit_per_min
                    else:
                        req_count -= 1

                prompt = get_prompt_from_csv(
                        ground_truth=ground_truth_as_examples if sink != 'logging' else logging_ground_truth,
                        sink=sink,
                        prompt_type=prompt_type,
                        query_spec_tuple=(spec, is_unusual),
                        text_file=None,)
                try:
                    response = openai.Completion.create(
                      model=model,
                      prompt=prompt,
                      temperature=0,
                      max_tokens=6,
                      top_p=1.0,
                      frequency_penalty=0.0,
                      presence_penalty=0.0,
                      # n=5, # return 5 completions
                      logprobs=5, # return the top-5 log probabilities for each token
                      stop=["###", "\n"]
                    )
                except openai.error.RateLimitError as e:
                    logging.error('Rate limit error: {}'.format(e))
                    logging.error('Sleep for a minute then try again...')
                    time.sleep(60)
                    req_count = rate_limit_per_min
                    response = openai.Completion.create(
                      model=model,
                      prompt=prompt,
                      temperature=0,
                      max_tokens=6,
                      top_p=1.0,
                      frequency_penalty=0.0,
                      presence_penalty=0.0,
                      # n=5, # return 5 completions
                      logprobs=5, # return the top-5 log probabilities for each token
                      stop=["###", "\n"]
                    )
                logging.debug(prompt)
                logging.debug(response)

                # Evaluation
                # prediction.text is the predicted label:
                # - for 'DirectPrediction':
                #   - for 'logging' sink: either 'sensitive data' or 'insensitive data'
                #   - for other sinks: either 'unexpected' or 'expected'
                # - for 'SimilarityCheck': either 'No' or 'Yes'
                # If neither label is output, then treat it as 'unexpected'
                if prompt_type == 'DirectPrediction':
                    unexpected_label = 'unexpected'
                    expected_label = 'expected'
                    if sink == 'logging':
                        unexpected_label = 'sensitive'
                        expected_label = 'insensitive'
                elif prompt_type == 'SimilarityCheck':
                    unexpected_label = 'No'
                    expected_label = 'Yes'
                output = response.choices[0]
                # all_token_logprobs shows the most likely choice for each token in the output sequence
                all_token_logprobs = output['logprobs']['top_logprobs']

                # Find first occurence of the label text in all_token_logprobs and
                # update result of each dataset
                pred_unusual = classify(all_token_logprobs, expected_label, unexpected_label)
                if spec in ground_truth_random:
                    update_result(result_dict['random set'], pred_unusual, is_unusual)
                if spec in secbench_ground_truth:
                    update_result(result_dict['SecBench.js'], pred_unusual, is_unusual)
                if spec in ground_truth_whole:
                    update_result(result_dict['whole set'], pred_unusual, is_unusual)
                if spec in logging_ground_truth:
                    update_result(result_dict['logging flows'], pred_unusual, is_unusual)
                logging.debug('whole set: %s', result_dict['whole set'])
                logging.debug('SecBench.js: %s', result_dict['SecBench.js'])
                logging.debug('random set: %s', result_dict['random set'])
                logging.debug('logging flows: %s', result_dict['logging flows'])
                # breakpoint()
        for dataset_name, result in result_dict.items():
            if dataset_name == 'logging flows' and sink != 'logging':
                continue
            print('Metrics of dataset:', dataset_name)
            compute_metrics(sink, result)
        # breakpoint()
