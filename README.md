# Spotify-Session-Skipping-Behaviour
An investigation on different behaviours during entire song listening sessions with regards to the users' session-based skipping activity. The analysis is performed on the [Spotify's Music Streaming Sessions (MSSD)](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge) Dataset.

This repository contains the source code for the approach outlined in the Short Paper _Identifying Skipping Behaviour Types in Music Streaming Sessions_, accepted at the 30th ACM International Conference on Information and Knowledge Management (CIKM2021).

## Python Packages
The required Python packages can be found in `requirements.txt`. Using a package manager such as `pip`, they can be easily installed as follows:

`pip3 install -r requirements.txt`

## Data Preparation
The main scripts (`experiment.py` and `analysis.py`) require the entire MSSD training dataset to be in a specific format. Specifically, they require every day as a SQLite database. To do so, the following instructions have to be followed (only once):

**Part 1**. In `data/training_set/`, sub-folders for every day have to be created. The following structure is expected:
```
data/training_set/20180715/
data/training_set/20180716/
data/training_set/20180717/
...
data/training_set/20180918/
```

**Part 2**. In each of these newly created subfolders, copy all original csv files for that day from the original dataset. This means that, for example, in folder `data/training_set/20180715/`, it is expected to have `log_0_20180715_000000000000.csv`, `log_1_20180715_000000000000.csv`, `log_2_20180715_000000000000.csv`, ..., `log_9_20180715_000000000000.csv`.

**Part 3**. Run `python data_preparation.py`, and all individual databases should be automatically created. Depending on the amount of selected data, this process may make some time.

## Run an Experiment
Having now completed the prior step (**Data Preparation**), it is now possible to run experiments. This can be done via the following command:

`python experiment.py --name All --type MyAllExperiment -l 20 --pca 7`

This will create an experiment in `results`, named `MyAllExperiment`, with an `All` experimental condition (meaning all sessions and on all days), for listening sessions of length 20, and with 7 PCA components. Further, individual boxplots for each skipping types is generated and available in the `figures` sub-folder.

The available experimental conditions flags are: _all_, _weekday_, _weekend_, _morning_, _afternoon_, _evening_, and _night_. Additionally, to perform an experiment on playlist types (e.g. editorial playlist), the array attribute `context_types` in `experiment.py` has to be modified accordingly. If empty (default), no playlist types filtering is applied when collecting listening sessions.

Finally, to modify the number of clusters, the `N_CLUSTERS` attribute in `constants.py` is to be changed.

## Perform Analysis
This last script allows for comparison of the identified types for experiments of a same session length via clusters matching. The metric used for matching clusters is the Euclidean distance. The analysis can be performed with the following command:

`python analysis.py`

Important to note is the fact that, when comparing distributions for different session lengths (via stacked histogram), it is required to manually rearrange the `distr_dict` rows to the desired sequence of skipping types. This is a necessary step if we want to correctly report distributions on different lengths and for the same sequence of types, such as "listener, listen-then-skip, skip-then-listen, skipper".
