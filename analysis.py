import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constants as consts

from itertools import tee
from sklearn.metrics.pairwise import euclidean_distances

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# load dataframe (with cluster labels) from path
def __load_day(path):
    
    df = pd.read_parquet("{0}/dataframe.parquet".format(path))

    kmeans_model = pickle.load(open("{0}/kmeans_models/kmeans_{1}.pkl".format(path, consts.N_CLUSTER_STR), "rb"))
    df["Cluster Number"] = kmeans_model.labels_

    return df

# this method returns a matrix representation of the day passed in `path`. If there are 4 clusters and the session length is 10,
# it contains 4 vectors of length 10. Each vector corresponds to the average session type of that cluster.
def __load_session_types_of_day(path):
    df = __load_day(path)

    # group by the cluster number and return mean aggregation, sorted by cluster number
    result_mean = df.groupby(["Cluster Number"], as_index=False).agg("mean").sort_values(by="Cluster Number", ascending=True)

    # at this point, vectors in `result_mean` are ordered, meaning the 1st row corresponds to Cluster 0, 2nd to Cluster 1, etc.
    # Drop the cluster number column
    result_mean.drop("Cluster Number", axis=1, inplace=True)

    # convert to numpy
    result_mean = result_mean.to_numpy()

    return result_mean

def __matchings_of_two_experiments(distance_matrix):
    # number of rows, which is equivalent to number of clusters
    n_rows = distance_matrix.shape[0]

    # `i` here also refers to the cluster number of Group A. The aim is to find a match in Group B, and put mapping
    # in `_dict` for later use
    _dict = {}
    for i in range(n_rows):
        # get current row
        row_arr = distance_matrix[i]

        # get index of minimum value
        h_index = np.where(row_arr == np.amin(row_arr))[0][0]

        _dict[i] = h_index

    return _dict

# this method loops through all experiments passed as input (`experiments`), and it returns a dictionary
# with all the matchings among clusters. For example, for the first tuple of experiments (A, B), it returns
# a mapping from clusters of A to clusters of B
def __create_similarity_matching(experiments):
    # loop all experiments pairwise
    _matching_dict = {}
    for exp_a, exp_b in pairwise(experiments):
        print("Matching: {0} - {1}".format(exp_a, exp_b))

        # load matrices for the two experiments under analysis
        df_a = __load_session_types_of_day(exp_a)
        df_b = __load_session_types_of_day(exp_b)

        # calculate euclidean distance between two matrices (each matrix has n vectors, where n is n_clusters)
        dis = euclidean_distances(df_a, df_b)

        # add matchings of ExpA and ExpB to overall dict
        _matching_dict[(exp_a, exp_b)] = __matchings_of_two_experiments(dis)

    return _matching_dict

def __generate_distributions_by_cluster(matching_dict):
    # create dictionary with individual distributions for each experiment
    _distr_dict = {}
    for (exp_a, _), _ in matching_dict.items():
        # we calculate the distribution in percentage only for exp_a. Eventually, all days will be looped through. This is why
        # the last day is duplicated
        df = __load_day(exp_a)
        _local_dict = df["Cluster Number"].value_counts(normalize=True).to_dict()
        _distr_dict[exp_a] = _local_dict

    # array that will hold all distribution values. 1st row is Cluster 0, 2nd row is Cluster 1, etc.
    distr_array = np.zeros(shape=(4, len(matching_dict)))
    for i in range(consts.N_CLUSTERS_INT):
        num_to_look_for = i
        j = 0
        for (exp_a, _), vals in matching_dict.items():
            distr_array[i, j] = _distr_dict[exp_a][num_to_look_for]

            # update num_to_look_for by following "translation"
            num_to_look_for = vals[num_to_look_for]
            j += 1

    return distr_array

def __distribution_stacked_histogram(distribution_dict, labels, base_path):
    filename = "{0}/stacked_distribution_histogram.png".format(base_path)

    fig, ax = plt.subplots()

    # use percentages and with 1 decimal points
    distribution_dict = np.round(distribution_dict * 100, 1)

    cmap = matplotlib.cm.get_cmap("tab20c")

    # Create gaps (filled by gray lines) in between bars
    y_pos = [0,1.25,2.25,3.5,4.5,5.5,6.5,7.75,8.75,9.75,10.75,11.75,12.75]

    # plot horizontal stacked bars
    ax.barh(y_pos, distribution_dict[0], color=cmap(0.5), label="listener")
    left = distribution_dict[0]
    ax.barh(y_pos, distribution_dict[1], left=left, color=cmap(0.15), label="listen-then-skip")
    left += distribution_dict[1]
    ax.barh(y_pos, distribution_dict[2], left=left, color=cmap(0.05),label="skip-then-listen")
    left += distribution_dict[2]
    ax.barh(y_pos, distribution_dict[3], left=left, color=cmap(0.35), label="skipper")

    ax.axhline(0.625, color="tab:gray")
    ax.axhline(2.875, color="tab:gray")
    ax.axhline(7.125, color="tab:gray")

    # add percentages in each bar
    for c in ax.containers:
        ax.bar_label(c, label_type="center")

    # change frequency of x-axis for 25-50-75-100
    plt.xticks(np.arange(0, 101, 25))
    plt.grid(axis="x")

    # set y-axis labels
    plt.yticks(y_pos, labels)

    # Long/Medium/Short label opposite of y-axis
    plt.gcf().text(1, 0.6, "Long Sessions", rotation=90, ha="center", va="center")

    # Add legend
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)

    ax.tick_params(axis="both", which="both")
    ax.invert_yaxis()

    # remove margin at the end of the x-axis (after 100)
    plt.margins(x=0)
    plt.margins(y=0.03)

    fig.tight_layout()
    fig.savefig(filename, dpi=fig.dpi, bbox_inches="tight")
    plt.close()

def main():
    # get list of experiments to agglomerate together
    base_path = "results/"
    experiments = sorted(glob.glob("{0}/*".format(base_path)))

    # repeat last element in the list of experiments. It is needed for the current matching implementation, otherwise the last day will be ignored!
    experiments.append(experiments[-1])

    print("Creating matchings dictionary")
    _matching_dict = __create_similarity_matching(experiments)

    print("\nGenerating Distribution of Clusters")
    distr_dict = __generate_distributions_by_cluster(_matching_dict)

    # labels for histogram to better handling of names
    labels = ["All", "Weekday", "Weekend", "Morning", "Afternoon", "Evening", "Night", "Editorial Playlist", "User Collection", "Catalog", "Radio", "Charts", "Personalized Playlist"]

    # Note: if we are creating multiple stacked histograms for later merging into one (e.g. long/medium/short sessions),
    # for each graph it is required to manually rearrange the `distr_dict` rows to the desired sequence of types.
    # This is necessary if we want to report the sequence "listener, listen-then-skip, skip-then-listen, skipper".
    # An example of this rearrange is:
    #distr_dict = distr_dict[[2, 3, 0, 1],:]

    __distribution_stacked_histogram(distr_dict, labels, base_path)

if __name__ == "__main__":
    main()