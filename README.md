# tensorflow-projects

This project ingests real world data from csv, trains a NN using tensorflow using low level mechanics, saves model, loads it, and evaluates it in a real world setting such a sequential prediction used as a simulator. Use the `config.yml` file to modify common hyperparameters.

## Setup

```bash
conda env update -f environment.yml
```

## Usage

Train and save supervised learning model
```bash
python learn_sim.py --train
```

Test using the withheld test set
```bash
python learn_sim.py --test
```

Select an episode number from the dataset for sequential prediction.
```bash
python learn_sim.py --eval <EP_#>
```