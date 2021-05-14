# Simmer Headline Analysis Demo

This repository is based on the work of the [Fake News Challenge](http://fakenewschallenge.org), and builds upon the baseline implementation produced for the challenge ([fnc1-baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline)).

* Author: Angus Redlarski Williams
* Github: @angusrw
* Email: angusrwilliams@gmail.com

### Overview

[Project Info](https://www.angusrw.com/work/simmer)

[<img src="https://static1.squarespace.com/static/5ddc2ba5a28c1a715fbec0cc/t/609eafc44460441b08923ab6/1621012429348/tech_posterjpg.jpg">](https://static1.squarespace.com/static/5ddc2ba5a28c1a715fbec0cc/t/6071ea9b7ffe752be9575e3e/1618078365622/tech_poster.pdf)

See [main.py](https://github.com/angusrw/simmer_demo/blob/master/main.py) for model implementation code.

See [feature_engineering.py](https://github.com/angusrw/simmer_demo/blob/master/feature_engineering.py) for NLP feature generation code.


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
