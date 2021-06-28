# Active Feature Acquisition with Generative Surrogate Models
## Get Started

### Prerequisites

refer to `requirements.txt`

### Download data

download your training data into the data folder. You might need to change the path for each dataset in `datasets` folder accordingly.

## Train and Test

You can train your own model by the scripts provided below.

### Cube

- Train the ACflow Classifier model

``` bash
python scripts/train.py --cfg_file=./exp/vec/params.json
```

- Train the PPO Policy

``` bash
python scripts/run_dfa.py --cfg_file=./exp/ppo/params.json
```
