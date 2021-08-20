# import libraries
import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import experiment_data_collection
import constants as consts

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def __get_dfs_for_boxplot(dataframe, cluster_number):
    sub_df = dataframe[dataframe["Cluster Number"] == cluster_number]
    result_mean = sub_df.groupby(["Cluster Number"], as_index=False).agg("mean").sort_values(by="Cluster Number", ascending=True)

    sub_df = sub_df.drop("Cluster Number", axis=1)
    result_mean = result_mean.drop("Cluster Number", axis=1)

    return sub_df, result_mean

def __generate_session_types_boxplots(dataframe, path):
    for i in range(consts.N_CLUSTERS_INT):
        os.makedirs(os.path.dirname("{0}/boxplots/".format(path)), exist_ok=True)
        # sub_df is a dataframe that contains only sessions of `i` cluster
        # result_mean is a single vector representation of that dataframe. Each session position (column) has a value which is the average of all positions
        sub_df, result_mean = __get_dfs_for_boxplot(dataframe, i)

        fig, ax = plt.subplots()

        # plot the boxplot
        plt.boxplot(sub_df,
            whiskerprops=dict(linestyle="--"),
            medianprops=dict(color="blue", linewidth=2),
            flierprops=dict(markersize=4))
        # plot the average red line
        plt.plot(list(range(1, SESSION_LENGTH + 1)), result_mean.T.values, linewidth=2, color="tab:red", zorder=3)

        ax.set_ylim([0.75, 5.25])

        # y-axis
        loc = ticker.MultipleLocator(base=1.0)
        ax.yaxis.set_major_locator(loc)
        plt.ylabel("Skipping Pattern (1-5)")

        # Frequency of ticks in x-axis
        x = np.arange(1, SESSION_LENGTH+1, 1)
        ax.set_xticks(x[::4])
        ax.set_xticklabels(x[::4], rotation=45)

        # x-axis label
        plt.xlabel("Session Position")

        ax.tick_params(axis="both", which="both")
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        fig.savefig("{0}/boxplots/{1}.png".format(path, i+1), bbox_inches='tight')
        plt.close()

def create_experiment():
    path = "results/{0}".format(EXPERIMENT_NAME)
    os.makedirs(os.path.dirname(path + "/"), exist_ok=True)
    return path

def save_configuration():
    _dict = {
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "EXPERIMENT_TYPE": EXPERIMENT_TYPE,
        "SESSION_LENGTH": SESSION_LENGTH,
        "CONTEXT_TYPES": CONTEXT_TYPES,
        "PCA_COMPONENTS": PCA_COMPONENTS
    }
    with open("{0}/conf.json".format(EXPERIMENT_NAME), "w") as fp:
        json.dump(_dict, fp)

def collect_dataframe():
    dataframe_path = "{0}/dataframe.parquet".format(EXPERIMENT_NAME)

    # if it already exists, load it in memory. Otherwise, generate it (it is also automatically saved)
    if os.path.isfile(dataframe_path):
        dataframe = pd.read_parquet(dataframe_path)
    else:
        dataframe = experiment_data_collection.generate_dataframe(EXPERIMENT_TYPE, SESSION_LENGTH, CONTEXT_TYPES, dataframe_path)

    return dataframe

def pca_run(dataframe):
    pca_model_path = "{0}/pca.pkl".format(EXPERIMENT_NAME)

    # if PCA file already exists, load it. Otherwise, apply PCA on dataframe and then save it
    if os.path.isfile(pca_model_path):
        pca = pickle.load(open(pca_model_path, "rb"))
        pca_vals = pca.transform(dataframe)
    else:
        # apply PCA on input dataframe
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(dataframe)
        pca_vals = pca.transform(dataframe)
        pickle.dump(pca, open(pca_model_path,"wb"))

    return pca_vals

def kmeans_clustering(pca_vals):
    kmeans_models_path = "{0}/kmeans_models".format(EXPERIMENT_NAME)

    # if kmeans sub-folder exists, load the model. Otherwise, generate and also save it
    if os.path.isdir(kmeans_models_path):
        model = pickle.load(open("{0}/kmeans_{1}.pkl".format(kmeans_models_path, consts.N_CLUSTER_STR), "rb"))
    else:
        os.makedirs(os.path.dirname(kmeans_models_path + "/"))

        print("Performing kmeans on {0} clusters".format(consts.N_CLUSTERS_INT))
        model = KMeans(n_clusters=consts.N_CLUSTERS_INT, init="k-means++", random_state=0)
        model.fit(pca_vals)

        filename = "{0}/kmeans_{1}.pkl".format(kmeans_models_path, consts.N_CLUSTER_STR)
        pickle.dump(model, open(filename, "wb"))

    return model

def generate_plots(dataframe, model, base_figures_path):
    # add cluster labels
    dataframe["Cluster Number"] = model.labels_

    # generate sub-folder with model's number of clusters
    _path = "{0}/{1}".format(base_figures_path, consts.N_CLUSTERS_INT)
    os.makedirs(os.path.dirname(_path + "/"), exist_ok=True)

    __generate_session_types_boxplots(dataframe, _path)

def main(experiment_name, experiment_type, session_length, pca_components, context_types):
    global EXPERIMENT_NAME, EXPERIMENT_TYPE, SESSION_LENGTH, PCA_COMPONENTS, CONTEXT_TYPES
    EXPERIMENT_NAME = experiment_name
    EXPERIMENT_TYPE = experiment_type
    SESSION_LENGTH = session_length
    PCA_COMPONENTS = pca_components
    CONTEXT_TYPES = context_types

    print("------------------------------------------------------")
    print("Experiment name: {0}".format(EXPERIMENT_NAME))
    EXPERIMENT_NAME = create_experiment()
    print("Experiment saved in folder: {0}".format(EXPERIMENT_NAME))
    print("------------------")
    print("*** Parameters ***")
    print("EXPERIMENT_TYPE: {0}".format(EXPERIMENT_TYPE))
    print("SESSION_LENGTH: {0}".format(SESSION_LENGTH))
    print("CONTEXT_TYPES: {0}".format(CONTEXT_TYPES))
    print("PCA_COMPONENTS: {0}".format(PCA_COMPONENTS))
    print("------------------------------------------------------")

    print("Saving configuration")
    save_configuration()

    print("Collecting data")
    df = collect_dataframe()
    print("... number of records:{0}".format(df.shape[0]))

    # create `figures` folder if it doesn't exist
    figures_path = EXPERIMENT_NAME + "/figures"
    os.makedirs(os.path.dirname(figures_path + "/"), exist_ok=True)

    print("Transforming data with PCA")
    scores_pca = pca_run(df)

    print("Generating k-means models")
    model = kmeans_clustering(scores_pca)

    print("Generating figures")
    generate_plots(df, model, figures_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument("--name", help="Name of the experiment", required=True)

    # experiment type
    choices = consts.EXPERIMENT_TYPES
    parser.add_argument("--type", choices=choices, help="Type of the experiment", required=True)

    # length of sessions in consideration (10-20)
    parser.add_argument("-l", help="Length of sessions (default 20)", default=20, type=int)

    # number of PCA Componenets
    parser.add_argument("--pca", help="Number of PCA Components (default 7)", default=7, type=int)

    # setting constants
    args = parser.parse_args()

    # there are 6 types of context: editorial_playlist, user_collection, catalog, radio, charts, personalized_playlist.
    # update `context_types` with wanted types (if 2+, a sorted array to avoid ordering issues), empty array otherwise
    context_types = []

    main(args.name, args.type, args.l, args.pca, context_types)
