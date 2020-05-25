# Simmer Headline Analysis Demo

This repository is based on the work of the [FakeChallenge.org](http://fakenewschallenge.org), and builds upon the baseline implementation produced for the challenge ([fnc1-baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline)).

Author - Angus Williams, github: @angusrw, email: angusrwilliams@gmail.com


## Getting Started
The FNC dataset is inlcuded as a submodule and can be FNC Dataset is included as a submodule. You should download the fnc-1 dataset by running the following commands. This places the fnc-1 dataset into the folder fnc-1/

    git submodule init
    git submodule update


The following libraries need to be installed before running any code:

    tqdm
    sklearn
    numpy
    scipy
    nltk
    xgboost

Before execution, `get_nltk.py` must be run once, in order to obtain the various datasets used in nltk methods:

    ``python3 get_nltk.py``

Running `main.py` will train an XGBoost model on the dataset over 10 folds:

    ``python3 main.py``

Features are stored after being generated to reduce time taken to run.

## Results

Score: 3560.5 out of 4448.5	(80.03821512869507%)

F1 score: [0.36655405 0.05649718 0.73708323 0.97205102]
