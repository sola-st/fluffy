# Fluffy: Bimodal Taint Analysis using CodeQL

Fluffy is a bimodal taint analysis that combines static analysis, which reasons
about data flow, with machine learning, which probabilistically determines which
flows are potentially problematic.

A typical use case would be to find unhygienic APIs, where values flow from
parameters of API functions exported by npm packages to vulnerable sinks, but a
user of the package might not expect this flow based on the name of the
parameter and/or the function.

For example, while it would not be surprising to find flow from a parameter
named `command` to a command-injection sink, it might be unexpected to find such
flow from a parameter named `fontFamily`.

Fluffy consists of two parts: a mining analysis implemented in CodeQL that
finds examples of flows from parameters to known sinks, and a machine-learning
component that, based on a corpus of such examples, learns how to distinguish
typical (and hence most likely unproblematic) flows from rare (and hence perhaps
unexpected) flows.

A detailed description of the approach is available in the paper [Beware of the
Unexpected: Bimodal Taint Analysis](https://arxiv.org/abs/2301.10545).

## Environment

Fluffy requires the following tools to be installed:

- Python (known to work with version 3.9)
- Node.js (known to work with version 16)
- Rust (known to work with version 1.65)
- CMake (known to work with version 3.23)
- CUDA (known to work with version 10.2)

## Installation

1. Ensure that the above tools are installed and available on your `PATH`.

2. Setup a virtual envionment for Fluffy by running `virtualenv -p python3.9 fluffy`, and activate it by running `source fluffy/bin/activate`.

3. Install Python dependencies by running `pip install --no-deps -r requirements.txt`.

4. If you have cloned this repository from GitHub, download the following data
   files from [figshare](https://figshare.com/s/1ab456424bfb5a2ead5e) and put
   them into the `data` directory: `flows-full.csv`, `train-set-2.csv`,
   `logging_flows_labelled.csv`, `logging_flows_train_set.csv`. You can do this
   with `wget`:

```sh
wget --no-check-certificate 'https://figshare.com/ndownloader/files/38176794?private_link=1ab456424bfb5a2ead5e' -O data/flows-full.csv
wget --no-check-certificate 'https://figshare.com/ndownloader/files/38176782?private_link=1ab456424bfb5a2ead5e' -O data/train-set-2.csv
wget --no-check-certificate 'https://figshare.com/ndownloader/files/38176776?private_link=1ab456424bfb5a2ead5e' -O data/logging_flows_labelled.csv
wget --no-check-certificate 'https://figshare.com/ndownloader/files/40649501?private_link=1ab456424bfb5a2ead5e' -O data/logging_flows_train_set.csv
```

5. Download and unzip the trained models from [figshare](https://figshare.com/s/1ab456424bfb5a2ead5e)
   (`m1_final.zip`, `m1_final_logging.zip`). Also, download the configuration file for the model (`detector_nn.pkl`).
   Again, you can do this with `wget`:

```sh
wget --no-check-certificate 'https://figshare.com/ndownloader/files/40467038?private_link=1ab456424bfb5a2ead5e' -O models/m1_final.zip
wget --no-check-certificate 'https://figshare.com/ndownloader/files/40467053?private_link=1ab456424bfb5a2ead5e' -O models/m1_final_logging.zip
unzip models/m1_final.zip -d models/
unzip models/m1_final_logging.zip -d models/
wget --no-check-certificate 'https://figshare.com/ndownloader/files/40649288?private_link=1ab456424bfb5a2ead5e' -O detector_nn.pkl
```

## Getting started

To quickly validate the general functionality, run the following command to run
the Novelty Detection experiment (Section 4.2):

For integrity violation:

```sh
python svm_outlier.py
```

For confidentiality violation:

```sh
python svm_outlier_logging.py
```

## Detailed Instructions

The following instructions detail how to run the code to validate the claims and
results in the paper.

### Sink Prediction

Train the model on the param-sink flows by running

```sh
./fluffy.py train -t neural data/train-set-2.csv --include-function-name --include-param-doc -w 
```

and on the logging flows by running

```sh
./fluffy.py train -t neural data/logging_flows_train_set.csv -w --logging-flow
```

Evaluate the model by on the param-sink flows running

```sh
./fluffy.py eval detector_nn.pkl --model-path models/m1_final/checkpoints/model-epoch\=00-val_loss\=0.596.ckpt 
```

Evaluate the model on the logging flows by running

```sh
./fluffy.py eval detector_nn.pkl --model-path models/m1_final_logging/checkpoints/model-epoch\=00-val_loss\=0.040.ckpt --logging-flow 
```

Replace the model path if you are training the model yourself. The trained models will be located in the `lightning_logs/` directory.

### Novelty Detection

Training and evaluation

```sh
python svm_outlier.py
python svm_outlier_logging.py
```

### Binary Classification

Training and evaluation (per sink type)

```sh
./fluffy.py finetune detector_nn.pkl data/test-set-2.csv CodeInjection
./fluffy.py finetune detector_nn.pkl data/test-set-2.csv CommandInjection 
...
./fluffy.py finetune detector_nn.pkl data/test-set-2.csv logging 
```

The experiment with data-set sizes can be reproduced by running

```sh
./fluffy.py finetune detector_nn.pkl data/test-set-2.csv CodeInjection -dse ; ./fluffy.py finetune detector_nn.pkl data/test-set-2.csv CommandInjection -dse ; ./fluffy.py finetune detector_nn.pkl data/test-set-2.csv ReflectedXss -dse ; ./fluffy.py finetune detector_nn.pkl data/test-set-2.csv TaintedPath -dse ; ./fluffy.py finetune detector_nn.pkl data/test-set-2.csv logging -dse
```

### Codex

Using the similarity-check prompt (per sink type)

```sh
python openai_query_api.py {your_key} --all_flows_examples --sink CommandInjection --prompt_type SimilarityCheck
...
```

Using the direct-prediction prompt (per sink type)

```
python openai_query_api.py {your_key} --all_flows_examples --sink CommandInjection --prompt_type DirectPrediction 
...
```

### Frequency-based

Training

```sh
./fluffy.py train -t counting -o detector_counting.pkl data/flows-full.csv 
```

Evaluation

```sh
./fluffy.py eval detector_counting.pkl
```

## Overview of implementation

### Mining analysis

The source code for the mining analysis is contained in `ql/queries`, with tests
in `ql/tests`.

- `Mining.ql`: Finds tuples of the form `(pkg, v, fn, p, k, d, f, l)`
  representing flow from parameter `p` of function `fn` in version `v` of
  package `pkg` to a sink of kind `k` (which is currently one of
  `CodeInjection`, `CommandInjection`, `ReflectedXss`, and `TaintedPath`), where
  the parameter is declared on line `l` of file `f`, and has doc comment `d`
  (which will be the empty string if not applicable).
- `Param2Sink.ql`: Can be used to explore the flow from a particular
  parameter to a particular sink by adjusting the part below the comment saying
  "Customise here".

The libraries `Param2Sink.qll` and `UMD.qll` contain shared logic used by both
queries.

### ML component

The ML component consists the classifiers for identifying unusual flows.

#### Classifiers

Classifiers can be trained and run using `fluffy.py`, as described above.

### Data

The `data` folder contains some manually-curated and some generated data:

- SecBench.js.csv: SecBench.js dataset (Section 4.1.3)
- ground-truth-full-flows.csv: random set (Section 4.1.3)
- ground-truth.csv: balanced set (Section 4.1.3)
- logging_flows_ground_truth.csv: ground truth for logging flows (Section 4.1.3)
- logging_flows_not_sensitive_unique.csv: subset of logging_flows_ground_truth.csv containing just the flows that were not considered sensitive by CodeQL
- survey_result.csv: results of the survey (Section 4.1.3)
- test-set-2.csv: labelled ground truth from ground-truth.csv
- flows-full.csv: param-sink flows (Section 4.1.1) (available [here](https://figshare.com/s/1ab456424bfb5a2ead5e?file=38176794))
- train-set-2.csv: flows-full.csv minus ground-truth.csv (available [here](https://figshare.com/s/1ab456424bfb5a2ead5e?file=38176782))
- logging_flows_labelled.csv: logging flows (Section 4.1.1) (available [here](https://figshare.com/s/1ab456424bfb5a2ead5e?file=38176776))
- logging_flows_train_set.csv: logging_flows_labelled.csv minus logging_flows_ground_truth.csv (available [here](https://figshare.com/s/1ab456424bfb5a2ead5e?file=40649501))

### Code

The following files contain the code for the ML component:

- classifier_net.py: deep-learning model (Sections 3.4.1, 3.4.2)
- compute_krippendorff.py: Section 4.1.3
- config.py: configuration
- flows.py: data representation of flows
- flows_dataset.py: Torch dataset for flows labelled with sink kind
- flows_unexpected_dataset.py: Torch dataset for flows labelled with expected/unpexpected
- openai_generate_prompt.py: generate prompt for LLM (Section 3.4.4)
- openai_query_api.py: driver script for calling the model
- quo_fluis.py: driver script
- svm_outlier.py: novelty detection model (Section 3.4.3) for integrity
- svm_outlier_logging.py: novelty detection model for confidentiality
- train_detector.py: code for training the detectors
- util.py: data conversion utilities
- util_plot.py: utilities for plotting ROC and PR curves
