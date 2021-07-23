# Active Feature Acquisition with Generative Surrogate Models
## Get Started

### Prerequisites

Refer to `requirements.txt`

### Download data and data preprocess

Download your training data into the data folder. You might need to convert the data file into a pickle file. The structure of the data should be a dictionary. The keys are 'train','valid', and 'test' and the values are the corresponding data tuple (x, y).
<br />
You might need to change the path for each dataset in `datasets` folder accordingly, in datasets folder, there is a corresponding file for each dataset that parse the data to fit the Tensorflow model.

## Train and Test

You can train your own model by the scripts provided below.

### Cube

- Train the ACflow model

``` bash
python scripts/train_model.py --cfg_file=./exp/cube/params.json
```

- Train the PPO Policy

``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/params.json
```
