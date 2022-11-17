#!/usr/bin/env python
# coding: utf-8

# ![STREAMLINE_LOGO.png](attachment:STREAMLINE_LOGO.png)

# # Summary

# This notebook runs all aspects of the STREAMLINE which is an automated machine learning analysis pipeline for binary classification tasks. Of note, two potentially important elements that are not automated by this pipeline include careful data cleaning and feature engineering using problem domain knowledge. Please review the README included in the associated GitHub repository for a detailed overview of how to run this pipeline. For simplicity, this notebook runs Python code outside of what is visible within it.
#
# This notebook is set up to run 'as-is' on a 'demo' dataset from the UCI repository (HCC dataset) using only three modeling algorithms (so that it runs in a matter of minutes). We analyze a copy of the dataset with and without covariate features to show how this pipline can be run on multiple datasets simultaneously (having the option to compare modeling on these different datasets in a later phase of the pipeline. Users will need to update pipeline run parameters below to ready the pipeline for their own needs. Suggested default run parameters suitible for most users are included, however file paths and names will need to be edited to run anything other than the 'demo' analysis.

# ## Notebook Housekeeping
# Set up notebook cells to display desired results. No need to edit.

# In[ ]:


import warnings
import sys
import os

warnings.filterwarnings("ignore")

# Jupyter Notebook Hack: This code ensures that the results of multiple commands within a given cell are all displayed, rather than just the last.
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# ## -----------------------------------------------------------------------------------------------------------------
# ## (User Specified) Run Parameters of STREAMLINE
# These initial notebook cells include all customizable run parameters for STREAMLINE. These settings should only be left unchanged for users wishing to test out the pipeline demo (as is) to learn how it works or to confirm efficacy before running their own data. Run parameters for each phase of the pipeline are included in separate code cells of this section of the notebook.
#

# ### Mandatory Run Parameters for Pipeline

# In[ ]:


demo_run = False  # Leave true to run the local demo dataset (without specifying any datapaths), make False to specify a different data folder path below

# Target dataset folder path(must include one or more .txt or .csv datasets)
data_path = "/home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/data_input"  # (str) Demontration Data Path Folder

# Output foder path: where to save pipeline outputs (must be updated for a given user)
output_path = "/home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/results"  # (str) Demonstration Ouput Path Folder
# output_path = 'C:/Users/UrbanowiczR/Documents/Analysis/STREAMLINE_Experiments' #User can ignore... For testing on my secondary PC

# Unique experiment name - folder created for this analysis within output folder path
experiment_name = "myoxia_streamline"  # (str) Demontration Experiment Name

# Data Labels
class_label = "conclusion"  # (str) i.e. class outcome column label
instance_label = "id"  # (str) If data includes instance labels, given respective column name here, otherwise put 'None'

# Option to manually specify feature names to leave out of analysis, or which to treat as categorical (without using built in variable type detector)
ignore_features = [
    "patient_id",
    "expert_id",
    "biopsie_id",
    "age_biopsie",
    "gene_diag",
    "muscle_prelev",
    "date_envoie",
    "mutation",
    "pheno_terms",
    "comment",
    "BOQA_prediction",
    "BOQA_prediction_score",
    "datetime",
]  # list of column names (given as string values) to exclude from the analysis (only insert column names if needed, otherwise leave empty)
categorical_feature_headers = (
    []
)  # empty list for 'auto-detect' otherwise list feature names (given as string values) to be treated as categorical. Only impacts algorithms that can take variable type into account.


# ### Run Parameters for Phase 1: Exploratory Analysis

# In[ ]:


cv_partitions = 2  # (int, > 1) Number of training/testing data partitions to create - and resulting number of models generated using each ML algorithm
partition_method = (
    "S"  # (str, S R or M) for stratified, random, or matched, respectively
)
match_label = "None"  # (str) Only applies when M selected for partition-method; indicates column label with matched instance ids'

categorical_cutoff = 10  # (int) Bumber of unique values after which a variable is considered to be quantitative vs categorical
sig_cutoff = 0.05  # (float, 0-1) Significance cutoff used throughout pipeline
export_feature_correlations = "True"  # (str, True or False) Run and export feature correlation analysis (yields correlation heatmap)
export_univariate_plots = "True"  # (str, True or False) Export univariate analysis plots (note: univariate analysis still output by default)
topFeatures = (
    10  # (int) Number of top features to report in notebook for univariate analysis
)
random_state = 42  # (int) Sets a specific random seed for reproducible results


# ### Run Parameters for Phase 2: Data Preprocessing

# In[ ]:


scale_data = "False"  # (str, True or False) Perform data scaling?
impute_data = "True"  # (str, True or False) Perform missing value data imputation? (required for most ML algorithms if missing data is present)
overwrite_cv = "True"  # (str, True or False) Overwrites earlier cv datasets with new scaled/imputed ones
multi_impute = "True"  # (str, True or False) Applies multivariate imputation to quantitative features, otherwise uses mean imputation


# ### Run Parameters for Phase 3: Feature Importance Evaluation

# In[ ]:


do_mutual_info = "True"  # (str, True or False) Do mutual information analysis
do_multisurf = "True"  # (str, True or False) Do multiSURF analysis
use_TURF = "False"  # (str, True or False) Use TURF wrapper around MultiSURF
TURF_pct = 0.5  # (float, 0.01-0.5) Proportion of instances removed in an iteration (also dictates number of iterations)
njobs = (
    -1
)  # (int) Number of cores dedicated to running algorithm; setting to -1 will use all available cores
instance_subset = 2000  # (int) Sample subset size to use with multiSURF


# ### Run Parameters for Phase 4: Feature Selection

# In[ ]:


max_features_to_keep = 2000  # (int) Maximum features to keep. 'None' if no max
filter_poor_features = "True"  # (str, True or False) Filter out the worst performing features prior to modeling
top_features = 40  # (int) Number of top features to illustrate in figures
export_scores = "True"  # (str, True or False) Export figure summarizing average feature importance scores over cv partitions


# ### Run Parameters for Phase 5: Modeling

# In[ ]:


# ML Model Algorithm Options (individual hyperparameter options can be adjusted below)
do_all = "False"  # (str, True or False) indicates default value for whether all or none of the algorithms should be run
do_NB = "True"  # (str, True or False, or None) Run naive bayes modeling
do_LR = "False"  # (str, True or False, or None) Run logistic regression modeling
do_DT = "True"  # (str, True or False, or None) Run decision tree modeling
do_RF = "False"  # (str, True or False, or None) Run random forest modeling
do_GB = "False"  # (str, True or False, or None) Run gradient boosting modeling
do_XGB = "False"  # (str, True or False, or None) Run XGBoost modeling
do_LGB = "False"  # (str, True or False, or None) Run LGBoost modeling
do_CGB = "False"  # (str, True or False, or None) Run Catboost modeling
do_SVM = "False"  # (str, True or False, or None) Run support vector machine modeling
do_ANN = "False"  # (str, True or False, or None) Run artificial neural network modeling
do_KNN = "False"  # (str, True or False, or None) Run k-neighbors classifier modeling
do_GP = "False"  # (str, True or False, or None) Run genetic programming symbolic classifier modeling

# ML Algorithms implemented by our reserach group: Rule-based ML Algorithm Options (Computationally expensive, so can be impractical to run hyperparameter sweep)
do_eLCS = "False"  # (str, True or False, or None) Run eLCS modeling (a basic supervised-learning learning classifier system)
do_XCS = "False"  # (str, True or False, or None) Run XCS modeling (a supervised-learning-only implementation of the best studied learning classifier system)
do_ExSTraCS = "False"  # (str, True or False, or None) Run ExSTraCS modeling (a learning classifier system designed for biomedical data mining)

# Other Analysis Parameters
training_subsample = 0  # (int) For long running algorithms, option to subsample training set (0 for no subsample) Limit Sample Size Used to train algorithms that do not scale up well in large instance spaces (i.e. XGB,SVM,KN,ANN,and LR to a lesser degree) and depending on 'instances' settings, ExSTraCS, eLCS, and XCS)
use_uniform_FI = "True"  # (str, True or False) Overides use of any available feature importances estimate methods from models, instead using permutation_importance uniformly
primary_metric = "accuracy"  # (str) Must be an available metric identifier from (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

# Hyperparameter Sweep Options
n_trials = 10  # (int or None) Number of bayesian hyperparameter optimization trials using optuna
timeout = 100  # (int or None) Seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started)
export_hyper_sweep_plots = (
    "True"  # (str, True or False) Export hyper parameter sweep plots from optuna
)

# Learning classifier system specific options (ExSTraCS, eLCS, XCS)
do_lcs_sweep = (
    "False"  # (str, True or False) Do LCS hyperparam tuning or use below params
)
nu = 1  # (int, 0-10) Fixed LCS nu param
iterations = 5000  # (int, > data sample size) Fixed LCS # learning iterations param
N = 100  # (int) > 500) Fixed LCS rule population maximum size param
lcs_timeout = 200  # (int) Seconds until hyperparameter sweep stops for LCS algorithms (evolutionary algorithms often require more time for a single run)


# ### Hyperparameter Sweep Options for ML Algorithms
# Users can extend or limit the range or options for given ML algorithm hyperparameters to be tested in hyperparameter optimization. These options are hardcoded when running this pipeline from the command line, but they are available here for users to see and modify. We have sought to include a broad range of relevant configurations based on online examples and relevant research publications. Use caution when modifying values below as improper modifications will lead to pipeline errors/failure. Links to available hyperparameter options for each algorithm are included below.

# In[ ]:


def hyperparameters(random_state, do_lcs_sweep, nu, iterations, N, feature_names):
    param_grid = {}
    # Naive Bayes - no hyperparameters

    # Logistic Regression (Note: can take longer to run in data with larger instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    param_grid_LR = {
        "penalty": ["l2", "l1"],
        "C": [1e-5, 1e5],
        "dual": [True, False],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "class_weight": [None, "balanced"],
        "max_iter": [10, 1000],
        "random_state": [random_state],
    }

    # Decision Tree
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree%20classifier#sklearn.tree.DecisionTreeClassifier
    param_grid_DT = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [1, 30],
        "min_samples_split": [2, 50],
        "min_samples_leaf": [1, 50],
        "max_features": [None, "auto", "log2"],
        "class_weight": [None, "balanced"],
        "random_state": [random_state],
    }

    # Random Forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier
    param_grid_RF = {
        "n_estimators": [10, 1000],
        "criterion": ["gini", "entropy"],
        "max_depth": [1, 30],
        "min_samples_split": [2, 50],
        "min_samples_leaf": [1, 50],
        "max_features": [None, "auto", "log2"],
        "bootstrap": [True],
        "oob_score": [False, True],
        "class_weight": [None, "balanced"],
        "random_state": [random_state],
    }

    # Gradient Boosting Trees
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradient%20boosting#sklearn.ensemble.GradientBoostingClassifier
    param_grid_GB = {
        "n_estimators": [10, 1000],
        "loss": ["log_loss"],
        "learning_rate": [0.0001, 0.3],
        "min_samples_leaf": [1, 50],
        "min_samples_split": [2, 50],
        "max_depth": [1, 30],
        "random_state": [random_state],
    }

    # XG Boost (Note: Not great for large instance spaces (limited completion) and class weight balance is included as option internally
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param_grid_XGB = {
        "booster": ["gbtree"],
        "objective": ["multi:softproba"],
        "verbosity": [0],
        "reg_lambda": [1e-8, 1.0],
        "alpha": [1e-8, 1.0],
        "eta": [1e-8, 1.0],
        "gamma": [1e-8, 1.0],
        "max_depth": [1, 30],
        "grow_policy": ["depthwise", "lossguide"],
        "n_estimators": [10, 1000],
        "min_samples_split": [2, 50],
        "min_samples_leaf": [1, 50],
        "subsample": [0.5, 1.0],
        "min_child_weight": [0.1, 10],
        "colsample_bytree": [0.1, 1.0],
        "nthread": [1],
        "seed": [random_state],
        "num_class": [num_classes],
    }

    # LG Boost (Note: class weight balance is included as option internally (still takes a while on large instance spaces))
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    param_grid_LGB = {
        "objective": ["multiclass"],
        "metric": [""],
        "verbose": [-1],
        "boosting_type": ["gbdt"],
        "num_leaves": [2, 256],
        "max_depth": [1, 30],
        "reg_alpha": [1e-8, 10.0],
        "reg_lambda": [1e-8, 10.0],
        "colsample_bytree": [0.4, 1.0],
        "subsample": [0.4, 1.0],
        "subsample_freq": [1, 7],
        "min_child_samples": [5, 100],
        "n_estimators": [10, 1000],
        "n_jobs": [1],
        "random_state": [random_state],
    }

    # CatBoost - (Note this is newly added, and further optimization to this configuration is possible)
    # https://catboost.ai/en/docs/references/training-parameters/
    param_grid_CGB = {
        "learning_rate": [0.0001, 0.3],
        "iterations": [10, 500],
        "depth": [1, 10],
        "l2_leaf_reg": [1, 9],
        "loss_function": ["MultiClass"],
        "random_seed": [random_state],
    }

    # Support Vector Machine (Note: Very slow in large instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    param_grid_SVM = {
        "kernel": ["linear", "poly", "rbf"],
        "C": [0.1, 1000],
        "gamma": ["scale"],
        "degree": [1, 6],
        "probability": [True],
        "class_weight": [None, "balanced"],
        "random_state": [random_state],
    }

    # Artificial Neural Network (Note: Slow in large instances spaces, and poor performer in small instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=artificial%20neural%20network
    param_grid_ANN = {
        "n_layers": [1, 3],
        "layer_size": [1, 100],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "momentum": [0.1, 0.9],
        "solver": ["sgd", "adam"],
        "batch_size": ["auto"],
        "alpha": [0.0001, 0.05],
        "max_iter": [200],
        "random_state": [random_state],
    }

    # K-Nearest Neighbor Classifier (Note: Runs slowly in data with large instance space)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier
    param_grid_KNN = {
        "n_neighbors": [1, 100],
        "weights": ["uniform", "distance"],
        "p": [1, 5],
        "metric": ["euclidean", "minkowski"],
    }

    # Genetic Programming Symbolic Classifier
    # https://gplearn.readthedocs.io/en/stable/reference.html
    param_grid_GP = {
        "population_size": [100, 1000],
        "generations": [10, 500],
        "tournament_size": [3, 50],
        "init_method": ["grow", "full", "half and half"],
        "function_set": [
            ["add", "sub", "mul", "div"],
            [
                "add",
                "sub",
                "mul",
                "div",
                "sqrt",
                "log",
                "abs",
                "neg",
                "inv",
                "max",
                "min",
            ],
            [
                "add",
                "sub",
                "mul",
                "div",
                "sqrt",
                "log",
                "abs",
                "neg",
                "inv",
                "max",
                "min",
                "sin",
                "cos",
                "tan",
            ],
        ],
        "parsimony_coefficient": [0.001, 0.01],
        "feature_names": [feature_names],
        "low_memory": [True],
        "random_state": [random_state],
    }

    # Learning Classifier Systems (i.e. eLCS, XCS, and ExSTraCS)
    # https://github.com/UrbsLab/scikit-eLCS
    # https://github.com/UrbsLab/scikit-XCS
    # https://github.com/UrbsLab/scikit-ExSTraCS

    if eval(do_lcs_sweep):
        # eLCS
        param_grid_eLCS = {
            "learning_iterations": [100000, 200000, 500000],
            "N": [1000, 2000, 5000],
            "nu": [1, 10],
            "random_state": [random_state],
        }
        # XCS
        param_grid_XCS = {
            "learning_iterations": [100000, 200000, 500000],
            "N": [1000, 2000, 5000],
            "nu": [1, 10],
            "random_state": [random_state],
        }
        # ExSTraCS
        param_grid_ExSTraCS = {
            "learning_iterations": [100000, 200000, 500000],
            "N": [1000, 2000, 5000],
            "nu": [1, 10],
            "random_state": [random_state],
            "rule_compaction": [None],
        }
    else:
        # eLCS
        param_grid_eLCS = {
            "learning_iterations": [iterations],
            "N": [N],
            "nu": [nu],
            "random_state": [random_state],
        }
        # XCS
        param_grid_XCS = {
            "learning_iterations": [iterations],
            "N": [N],
            "nu": [nu],
            "random_state": [random_state],
        }
        # ExSTraCS
        param_grid_ExSTraCS = {
            "learning_iterations": [iterations],
            "N": [N],
            "nu": [nu],
            "random_state": [random_state],
            "rule_compaction": ["QRF"],
        }  # 'None','QRF' - which is quick rule filter

    # Leave code below as is...
    param_grid["Naive Bayes"] = {}
    param_grid["Logistic Regression"] = param_grid_LR
    param_grid["Decision Tree"] = param_grid_DT
    param_grid["Random Forest"] = param_grid_RF
    param_grid["Gradient Boosting"] = param_grid_GB
    param_grid["Extreme Gradient Boosting"] = param_grid_XGB
    param_grid["Light Gradient Boosting"] = param_grid_LGB
    param_grid["Category Gradient Boosting"] = param_grid_CGB
    param_grid["Support Vector Machine"] = param_grid_SVM
    param_grid["Artificial Neural Network"] = param_grid_ANN
    param_grid["K-Nearest Neightbors"] = param_grid_KNN
    param_grid["Genetic Programming"] = param_grid_GP
    param_grid["eLCS"] = param_grid_eLCS
    param_grid["XCS"] = param_grid_XCS
    param_grid["ExSTraCS"] = param_grid_ExSTraCS
    return param_grid


# ### Run Parameters for Phase 6:  Statistics Summary and Figure Generation

# In[ ]:


plot_ROC = "False"  # (str, True or False) Plot ROC curves individually for each algorithm including all CV results and averages
plot_PRC = "False"  # (str, True or False) Plot PRC curves individually for each algorithm including all CV results and averages
plot_FI_box = "False"  # (str, True or False) Plot box plot summaries comparing algorithms for each metric
plot_metric_boxplots = (
    "False"  # (str, True or False) Plot feature importance boxplots for each algorithm
)
metric_weight = "balanced_accuracy"  # (str, balanced_accuracy or roc_auc) ML model metric used as weight in composite FI plots (only supports balanced_accuracy or roc_auc as options) Recommend setting the same as primary_metric if possible.
top_model_features = (
    10  # (int) Number of top features in model to illustrate in figures
)


# ### Run Parameters for Phase 10:  Apply Models to Replication Dataset
# An optional phase to apply all trained models from previous phases to a separate 'replication' dataset which will be used to evaluate models across all algorithms and CV splits. In this demo, we didn't have a separate replication dataset to use for the UCI HCC dataset evaluated. Thus here we use a copy of the original HCC dataset as a 'pretend' replication dataset to demonstrate functionality. The replication data folder can include 1 or more datasets that can be evaluated as separate replication data. The user also needs to

# In[ ]:


applyToReplication = False  # (Boolean, True or False) Leave false unless you have a replication dataset handy to further evaluate/compare all models in uniform manner
rep_data_path = ""  # (txt) Name of folder with replication Dataset(s)
dataset_for_rep = ""  # (txt) Path and name of dataset used to generate the models we want to apply (not the replication dataset)


# ### Run Parameters for Phase 11:  File Cleanup
# An optional phase to delete all unnecessary/temporary files generated by the pipeline.

# In[ ]:


del_time = (
    "True"  # (str, True or False) Delete individual run-time files (but save summary)
)
del_oldCV = "True"  # (str, True or False) Delete any of the older versions of CV training and testing datasets not overwritten (preserves final training and testing datasets)


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 1: Exploratory Analysis

# ### Identify Working Directory

# In[ ]:


wd_path = os.getcwd()  # Working directory path automatically detected
# wd_path = wd_path.replace('','/')
sys.path.insert(-1, wd_path + "/streamline")


# ### Import Python Packages

# In[ ]:


import glob
import time
import csv
import pandas as pd
import numpy as np
import random
import pickle
import ExploratoryAnalysisMain
import ExploratoryAnalysisJob


# ### Demo Setup
# Bypasses whatever user may have entered into 'data_path' variable to ensure proper loading of local 'demo' dataset.

# In[ ]:


if demo_run:
    data_path = wd_path + "/DemoData"
print("Data Folder Path: " + data_path)
jupyterRun = "True"  # Leave True or pipeline will not display text or figures


# ### Run Exploratory Analysis

# In[ ]:


ExploratoryAnalysisMain.makeDirTree(data_path, output_path, experiment_name, jupyterRun)


# ### Phase 1.1: Clean our dataset and encode the values

# In[ ]:


from sklearn import preprocessing

raw_df = pd.read_csv("EHRoes/text_reports.csv")

# Modify Onto features values: on modifie les valeur: 0:-1, puis on modifie les N/A en 0
# Get list of ontology columns
onto_col = []
for column in raw_df.columns.to_list():
    if column[0:4] == "MHO:":
        onto_col.append(column)

#  Simplify quantity
raw_df[onto_col] = raw_df[onto_col].replace({0: 0, 0.25: 1, 0.5: 1, 0.75: 1})

# One-hot Encode Cat Variable
# Needed if gene_diag included in analysis
# raw_df = pd.get_dummies(raw_df, columns=["gene_diag"])

# Label Encode Conclusion
raw_df["conclusion"].replace(
    {
        "COM_CCD": "COM",
        "UNCLEAR": np.NaN,
        "NM_CAP": "NM",
        "CFTD": np.NaN,
        "COM_MMM": "COM",
        "NON_CM": np.NaN,
        "CM": np.NaN,
    },
    inplace=True,
)
le = preprocessing.LabelEncoder()
fit_by = pd.Series([i for i in raw_df["conclusion"].unique() if type(i) == str])
le.fit(fit_by)
raw_df["conclusion"] = raw_df["conclusion"].map(
    lambda x: le.transform([x])[0] if type(x) == str else x
)
print("Class Label: ")
for index, value in enumerate(le.classes_):
    print("Class Value: ", index, " Class Name: ", value)

# Delete columns with all missing values
raw_df[onto_col].dropna(how="all", axis=1, inplace=True)

raw_df[onto_col] = raw_df[onto_col].fillna(0)

raw_df.to_csv(data_path + "/input.csv", sep=",", index=False)

# XGBoost N Class
classes = np.unique(raw_df[class_label].dropna())
num_classes = len(classes)


# In[ ]:


# Determine file extension of datasets in target folder:
file_count = 0
unique_datanames = []
for dataset_path in glob.glob(data_path + "/*"):
    # dataset_path = str(dataset_path).replace('','/')
    print(
        "---------------------------------------------------------------------------------"
    )
    print(dataset_path)
    file_extension = dataset_path.split("/")[-1].split(".")[-1]
    data_name = dataset_path.split("/")[-1].split(".")[
        0
    ]  # Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
    if file_extension == "txt" or file_extension == "csv":
        if data_name not in unique_datanames:
            unique_datanames.append(data_name)
            ExploratoryAnalysisJob.runExplore(
                dataset_path,
                output_path + "/" + experiment_name,
                cv_partitions,
                partition_method,
                categorical_cutoff,
                export_feature_correlations,
                export_univariate_plots,
                class_label,
                instance_label,
                match_label,
                random_state,
                ignore_features,
                categorical_feature_headers,
                sig_cutoff,
                jupyterRun,
            )
            file_count += 1

if file_count == 0:  # Check that there was at least 1 dataset
    raise Exception(
        "There must be at least one .txt or .csv dataset in data_path directory"
    )

# Create metadata dictionary object to keep track of pipeline run paramaters throughout phases
metadata = {}
metadata["Data Path"] = data_path
metadata["Output Path"] = output_path
metadata["Experiment Name"] = experiment_name
metadata["Class Label"] = class_label
metadata["Instance Label"] = instance_label
metadata["Ignored Features"] = ignore_features
metadata["Specified Categorical Features"] = categorical_feature_headers
metadata["CV Partitions"] = cv_partitions
metadata["Partition Method"] = partition_method
metadata["Match Label"] = match_label
metadata["Categorical Cutoff"] = categorical_cutoff
metadata["Statistical Significance Cutoff"] = sig_cutoff
metadata["Export Feature Correlations"] = export_feature_correlations
metadata["Export Univariate Plots"] = export_univariate_plots
metadata["Random Seed"] = random_state
metadata["Run From Jupyter Notebook"] = jupyterRun
# Pickle the metadata for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 2: Data Preprocessing

# ### Import Additional Python Packages

# In[ ]:


import DataPreprocessingJob


# ### Run Data Preprocessing

# In[ ]:


dataset_paths = os.listdir(output_path + "/" + experiment_name)
dataset_paths.remove("metadata.pickle")
for dataset_directory_path in dataset_paths:
    full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
    for cv_train_path in glob.glob(full_path + "/CVDatasets/*Train.csv"):
        # cv_train_path = str(cv_train_path).replace('','/')
        cv_test_path = cv_train_path.replace("Train.csv", "Test.csv")
        DataPreprocessingJob.job(
            cv_train_path,
            cv_test_path,
            output_path + "/" + experiment_name,
            scale_data,
            impute_data,
            overwrite_cv,
            categorical_cutoff,
            class_label,
            instance_label,
            random_state,
            multi_impute,
            jupyterRun,
        )

# Unpickle metadata from previous phase
file = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "rb")
metadata = pickle.load(file)
file.close()

# Update metadata
metadata["Use Data Scaling"] = scale_data
metadata["Use Data Imputation"] = impute_data
metadata["Use Multivariate Imputation"] = multi_impute
# Pickle the metadata for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 3: Feature Importance Evaluation

# ### Import Additional Python Packages

# In[ ]:


import FeatureImportanceJob


# ### Run Feature Importance Evaluation

# In[ ]:


dataset_paths = os.listdir(output_path + "/" + experiment_name)
removeList = removeList = [
    "metadata.pickle",
    "metadata.csv",
    "algInfo.pickle",
    "jobsCompleted",
    "logs",
    "jobs",
    "DatasetComparisons",
    "UsefulNotebooks",
    experiment_name + "_ML_Pipeline_Report.pdf",
]
for text in removeList:
    if text in dataset_paths:
        dataset_paths.remove(text)

for dataset_directory_path in dataset_paths:
    full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
    experiment_path = output_path + "/" + experiment_name

    if eval(do_mutual_info) or eval(do_multisurf):
        if not os.path.exists(full_path + "/feature_selection"):
            os.mkdir(full_path + "/feature_selection")

    if eval(do_mutual_info):
        if not os.path.exists(full_path + "/feature_selection/mutualinformation"):
            os.mkdir(full_path + "/feature_selection/mutualinformation")
        for cv_train_path in glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"):
            # cv_train_path = str(cv_train_path).replace('','/')
            FeatureImportanceJob.job(
                cv_train_path,
                experiment_path,
                random_state,
                class_label,
                instance_label,
                instance_subset,
                "mi",
                njobs,
                use_TURF,
                TURF_pct,
                jupyterRun,
            )

    if eval(do_multisurf):
        if not os.path.exists(full_path + "/feature_selection/multisurf"):
            os.mkdir(full_path + "/feature_selection/multisurf")
        for cv_train_path in glob.glob(full_path + "/CVDatasets/*_CV_*Train.csv"):
            # cv_train_path = str(cv_train_path).replace('','/')
            FeatureImportanceJob.job(
                cv_train_path,
                experiment_path,
                random_state,
                class_label,
                instance_label,
                instance_subset,
                "ms",
                njobs,
                use_TURF,
                TURF_pct,
                jupyterRun,
            )

# Unpickle metadata from previous phase
file = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "rb")
metadata = pickle.load(file)
file.close()

# Update metadata
metadata["Use Mutual Information"] = do_mutual_info
metadata["Use MultiSURF"] = do_multisurf
metadata["Use TURF"] = use_TURF
metadata["TURF Cutoff"] = TURF_pct
metadata["MultiSURF Instance Subset"] = instance_subset
# Pickle the metadata for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 4: Feature Selection

# ### Import Additional Python Packages

# In[ ]:


import FeatureSelectionJob


# ### Run Feature Selection

# In[ ]:


dataset_paths = os.listdir(output_path + "/" + experiment_name)
removeList = removeList = [
    "metadata.pickle",
    "metadata.csv",
    "algInfo.pickle",
    "jobsCompleted",
    "logs",
    "jobs",
    "DatasetComparisons",
    "UsefulNotebooks",
    experiment_name + "_ML_Pipeline_Report.pdf",
]
for text in removeList:
    if text in dataset_paths:
        dataset_paths.remove(text)

for dataset_directory_path in dataset_paths:
    full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
    FeatureSelectionJob.job(
        full_path,
        do_mutual_info,
        do_multisurf,
        max_features_to_keep,
        filter_poor_features,
        top_features,
        export_scores,
        class_label,
        instance_label,
        cv_partitions,
        overwrite_cv,
        jupyterRun,
    )

# Unpickle metadata from previous phase
file = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "rb")
metadata = pickle.load(file)
file.close()

# Update metadata
metadata["Max Features to Keep"] = max_features_to_keep
metadata["Filter Poor Features"] = filter_poor_features
metadata["Top Features to Display"] = top_features
metadata["Export Feature Importance Plot"] = export_scores
metadata["Overwrite CV Datasets"] = overwrite_cv
# Pickle the metadata for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 5: ML Modeling

# ### Phase 5 Import Additional Python Packages

# In[ ]:


import ModelJob


# In[ ]:


# Create ML modeling algorithm information dictionary, given as ['algorithm used (set to true initially by default)','algorithm abreviation', 'color used for algorithm on figures']
### Note that other named colors used by matplotlib can be found here: https://matplotlib.org/3.5.0/_images/sphx_glr_named_colors_003.png
### Make sure new ML algorithm abbreviations and color designations are unique
algInfo = {}
algInfo["Naive Bayes"] = [True, "NB", "silver"]
algInfo["Logistic Regression"] = [True, "LR", "dimgrey"]
algInfo["Decision Tree"] = [True, "DT", "yellow"]
algInfo["Random Forest"] = [True, "RF", "blue"]
algInfo["Gradient Boosting"] = [True, "GB", "cornflowerblue"]
algInfo["Extreme Gradient Boosting"] = [True, "XGB", "cyan"]
algInfo["Light Gradient Boosting"] = [True, "LGB", "pink"]
algInfo["Category Gradient Boosting"] = [True, "CGB", "magenta"]
algInfo["Support Vector Machine"] = [True, "SVM", "orange"]
algInfo["Artificial Neural Network"] = [True, "ANN", "red"]
algInfo["K-Nearest Neightbors"] = [True, "KNN", "chocolate"]
algInfo["Genetic Programming"] = [True, "GP", "purple"]
algInfo["eLCS"] = [True, "eLCS", "green"]
algInfo["XCS"] = [True, "XCS", "olive"]
algInfo["ExSTraCS"] = [True, "ExSTraCS", "lawngreen"]
### Add new algorithms here...

# Set up ML algorithm True/False use
if not eval(do_all):  # If do all algorithms is false
    for key in algInfo:
        algInfo[key][0] = False  # Set algorithm use to False

# Set algorithm use truth for each algorithm specified by user (i.e. if user specified True/False for a specific algorithm)
if not do_NB == "None":
    algInfo["Naive Bayes"][0] = eval(do_NB)
if not do_LR == "None":
    algInfo["Logistic Regression"][0] = eval(do_LR)
if not do_DT == "None":
    algInfo["Decision Tree"][0] = eval(do_DT)
if not do_RF == "None":
    algInfo["Random Forest"][0] = eval(do_RF)
if not do_GB == "None":
    algInfo["Gradient Boosting"][0] = eval(do_GB)
if not do_XGB == "None":
    algInfo["Extreme Gradient Boosting"][0] = eval(do_XGB)
if not do_LGB == "None":
    algInfo["Light Gradient Boosting"][0] = eval(do_LGB)
if not do_CGB == "None":
    algInfo["Category Gradient Boosting"][0] = eval(do_CGB)
if not do_SVM == "None":
    algInfo["Support Vector Machine"][0] = eval(do_SVM)
if not do_ANN == "None":
    algInfo["Artificial Neural Network"][0] = eval(do_ANN)
if not do_KNN == "None":
    algInfo["K-Nearest Neightbors"][0] = eval(do_KNN)
if not do_GP == "None":
    algInfo["Genetic Programming"][0] = eval(do_GP)
if not do_eLCS == "None":
    algInfo["eLCS"][0] = eval(do_eLCS)
if not do_XCS == "None":
    algInfo["XCS"][0] = eval(do_XCS)
if not do_ExSTraCS == "None":
    algInfo["ExSTraCS"][0] = eval(do_ExSTraCS)
### Add new algorithms here...

# Pickle the algorithm information dictionary for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "algInfo.pickle", "wb")
pickle.dump(algInfo, pickle_out)
pickle_out.close()

# Make list of algorithms to be run (full names)
algorithms = []
for key in algInfo:
    if algInfo[key][0]:  # Algorithm is true
        algorithms.append(key)


# ### Run ML Modeling

# In[ ]:


dataset_paths = os.listdir(output_path + "/" + experiment_name)
removeList = removeList = [
    "metadata.pickle",
    "metadata.csv",
    "algInfo.pickle",
    "jobsCompleted",
    "logs",
    "jobs",
    "DatasetComparisons",
    "UsefulNotebooks",
    experiment_name + "_ML_Pipeline_Report.pdf",
]
for text in removeList:
    if text in dataset_paths:
        dataset_paths.remove(text)
for dataset_directory_path in dataset_paths:
    full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
    if not os.path.exists(full_path + "/models"):
        os.mkdir(full_path + "/models")
    if not os.path.exists(full_path + "/model_evaluation"):
        os.mkdir(full_path + "/model_evaluation")
    if not os.path.exists(full_path + "/models/pickledModels"):
        os.mkdir(full_path + "/models/pickledModels")

    for cvCount in range(cv_partitions):
        train_file_path = (
            full_path
            + "/CVDatasets/"
            + dataset_directory_path
            + "_CV_"
            + str(cvCount)
            + "_Train.csv"
        )
        test_file_path = (
            full_path
            + "/CVDatasets/"
            + dataset_directory_path
            + "_CV_"
            + str(cvCount)
            + "_Test.csv"
        )
        for algorithm in algorithms:
            algAbrev = algInfo[algorithm][1]
            # Get header names for current CV dataset for use later in GP tree visulaization
            data_name = full_path.split("/")[-1]
            feature_names = pd.read_csv(
                full_path
                + "/CVDatasets/"
                + data_name
                + "_CV_"
                + str(cvCount)
                + "_Test.csv"
            ).columns.values.tolist()
            if instance_label != "None":
                feature_names.remove(instance_label)
            feature_names.remove(class_label)
            # Get hyperparameter grid
            param_grid = hyperparameters(
                random_state, do_lcs_sweep, nu, iterations, N, feature_names
            )[algorithm]
            ModelJob.runModel(
                algorithm,
                train_file_path,
                test_file_path,
                full_path,
                n_trials,
                timeout,
                lcs_timeout,
                export_hyper_sweep_plots,
                instance_label,
                class_label,
                random_state,
                cvCount,
                filter_poor_features,
                do_lcs_sweep,
                nu,
                iterations,
                N,
                training_subsample,
                use_uniform_FI,
                primary_metric,
                param_grid,
                algAbrev,
            )

# Unpickle metadata from previous phase
file = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "rb")
metadata = pickle.load(file)
file.close()

# Update metadata
metadata["Naive Bayes"] = str(algInfo["Naive Bayes"][0])
metadata["Logistic Regression"] = str(algInfo["Logistic Regression"][0])
metadata["Decision Tree"] = str(algInfo["Decision Tree"][0])
metadata["Random Forest"] = str(algInfo["Random Forest"][0])
metadata["Gradient Boosting"] = str(algInfo["Gradient Boosting"][0])
metadata["Extreme Gradient Boosting"] = str(algInfo["Extreme Gradient Boosting"][0])
metadata["Light Gradient Boosting"] = str(algInfo["Light Gradient Boosting"][0])
metadata["Category Gradient Boosting"] = str(algInfo["Category Gradient Boosting"][0])
metadata["Support Vector Machine"] = str(algInfo["Support Vector Machine"][0])
metadata["Artificial Neural Network"] = str(algInfo["Artificial Neural Network"][0])
metadata["K-Nearest Neightbors"] = str(algInfo["K-Nearest Neightbors"][0])
metadata["Genetic Programming"] = str(algInfo["Genetic Programming"][0])
metadata["eLCS"] = str(algInfo["eLCS"][0])
metadata["XCS"] = str(algInfo["XCS"][0])
metadata["ExSTraCS"] = str(algInfo["ExSTraCS"][0])
### Add new algorithms here...
metadata["Primary Metric"] = primary_metric
metadata["Training Subsample for KNN,ANN,SVM,and XGB"] = training_subsample
metadata["Uniform Feature Importance Estimation (Models)"] = use_uniform_FI
metadata["Hyperparameter Sweep Number of Trials"] = n_trials
metadata["Hyperparameter Sweep Number of Trials"] = n_trials
metadata["Hyperparameter Timeout"] = timeout
metadata["Export Hyperparameter Sweep Plots"] = export_hyper_sweep_plots
metadata["Do LCS Hyperparameter Sweep"] = do_lcs_sweep
metadata["LCS Hyperparameter: nu"] = nu
metadata["LCS Hyperparameter: Training Iterations"] = iterations
metadata["LCS Hyperparameter: N - Rule Population Size"] = N
metadata["LCS Hyperparameter Sweep Timeout"] = lcs_timeout
# Pickle the metadata for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 6: Statistics (Stats Summaries, Figures, Statistical Comparisons)

# ### Import Additional Python Packages

# In[ ]:


import StatsJob


# ### Run Statistics Summary and Figure Generation

# In[ ]:


# Unpickle metadata from previous phase
file = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "rb")
metadata = pickle.load(file)
file.close()
metadata["Export ROC Plot"] = plot_ROC
metadata["Export PRC Plot"] = plot_PRC
metadata["Export Metric Boxplots"] = plot_metric_boxplots
metadata["Export Feature Importance Boxplots"] = plot_FI_box
metadata["Metric Weighting Composite FI Plots"] = metric_weight
metadata["Top Model Features To Display"] = top_model_features
# Pickle the metadata for future use
pickle_out = open(output_path + "/" + experiment_name + "/" + "metadata.pickle", "wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()

# Now that primary pipeline phases are complete generate a human readable version of metadata
df = pd.DataFrame.from_dict(metadata, orient="index")
df.to_csv(output_path + "/" + experiment_name + "/" + "metadata.csv", index=True)

# Iterate through datasets
dataset_paths = os.listdir(output_path + "/" + experiment_name)
removeList = removeList = [
    "metadata.pickle",
    "metadata.csv",
    "algInfo.pickle",
    "jobsCompleted",
    "logs",
    "jobs",
    "DatasetComparisons",
    "UsefulNotebooks",
    experiment_name + "_ML_Pipeline_Report.pdf",
]
for text in removeList:
    if text in dataset_paths:
        dataset_paths.remove(text)
for dataset_directory_path in dataset_paths:
    full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
    StatsJob.job(
        full_path,
        plot_ROC,
        plot_PRC,
        plot_FI_box,
        class_label,
        instance_label,
        cv_partitions,
        scale_data,
        plot_metric_boxplots,
        primary_metric,
        top_model_features,
        sig_cutoff,
        metric_weight,
        jupyterRun,
    )


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 7: Dataset Comparison (Optional: Use only if > 1 dataset was analyzed)

# ### Import Additional Python Packages

# In[ ]:


import DataCompareJob


# ### Run Dataset Comparison

# In[ ]:


# if len(dataset_paths) > 1:
#     DataCompareJob.job(output_path+'/'+experiment_name,sig_cutoff,jupyterRun)


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 8: PDF Training Report Generator (Optional)

# In[ ]:


import PDF_ReportJob


# In[ ]:


experiment_path = output_path + "/" + experiment_name
PDF_ReportJob.job(experiment_path, "True", "None", "None")


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 9: Apply Models to Replication Data (Optional)

# ### Import Additional Python Packages

# In[ ]:


import ApplyModelJob


# ### Specify Run Parameters

# In[ ]:


# if demo_run:
#     rep_data_path = wd_path+'/DemoRepData'
#     dataset_for_rep = wd_path+'/DemoData/hcc-data_example.csv'
# print("Replication Data Folder Path: "+rep_data_path)
# print("Dataset Path: "+dataset_for_rep)


# ### Run Application of Models to Replication Data

# In[ ]:


# if applyToReplication:
#     data_name = dataset_for_rep.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
#     full_path = output_path + "/" + experiment_name + "/" + data_name #location of folder containing models respective training dataset

#     if not os.path.exists(full_path+"/applymodel"):
#         os.mkdir(full_path+"/applymodel")

#     #Determine file extension of datasets in target folder:
#     file_count = 0
#     unique_datanames = []
#     for datasetFilename in glob.glob(rep_data_path+'/*'):
#         datasetFilename = str(datasetFilename).replace('','/')

#         file_extension = datasetFilename.split('/')[-1].split('.')[-1]
#         apply_name = datasetFilename.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
#         if not os.path.exists(full_path+"/applymodel/"+apply_name):
#             os.mkdir(full_path+"/applymodel/"+apply_name)

#         if file_extension == 'txt' or file_extension == 'csv':
#             if apply_name not in unique_datanames:
#                 unique_datanames.append(apply_name)
#                 ApplyModelJob.job(datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,primary_metric,dataset_for_rep,match_label,plot_ROC,plot_PRC,plot_metric_boxplots,export_feature_correlations,jupyterRun,multi_impute)
#                 file_count += 1

#     if file_count == 0: #Check that there was at least 1 dataset
#         raise Exception("There must be at least one .txt or .csv dataset in rep_data_path directory")


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 10: PDF Apply Report Generator (Optional)

# In[ ]:


import PDF_ReportJob


# In[ ]:


# if applyToReplication:
#     experiment_path = output_path+'/'+experiment_name
#     PDF_ReportJob.job(experiment_path,'False',rep_data_path,dataset_for_rep)


# ## -----------------------------------------------------------------------------------------------------------------
# ## Phase 11: File Cleanup (Optional)

# In[ ]:


import shutil


# In[ ]:


# Get dataset paths for all completed dataset analyses in experiment folder
datasets = os.listdir(experiment_path)
experiment_name = experiment_path.split("/")[-1]  # Name of experiment folder
removeList = removeList = [
    "metadata.pickle",
    "metadata.csv",
    "algInfo.pickle",
    "jobsCompleted",
    "logs",
    "jobs",
    "DatasetComparisons",
    "UsefulNotebooks",
    experiment_name + "_ML_Pipeline_Report.pdf",
]
for text in removeList:
    if text in datasets:
        datasets.remove(text)

# Delete jobscompleted folder/files
try:
    shutil.rmtree(experiment_path + "/" + "jobsCompleted")
except:
    pass

# Delete target files within each dataset subfolder
for dataset in datasets:
    # Delete individual runtime files (save runtime summary generated in phase 6)
    if eval(del_time):
        try:
            shutil.rmtree(experiment_path + "/" + dataset + "/" + "runtime")
            print("Individual Runtime Files Deleted")
        except:
            pass
    # Delete temporary feature importance pickle files (only needed for phase 4 and then saved as summary files in phase 6)
    try:
        shutil.rmtree(
            experiment_path
            + "/"
            + dataset
            + "/feature_selection/mutualinformation/pickledForPhase4"
        )
        print("Mutual Information Pickle Files Deleted")
    except:
        pass
    try:
        shutil.rmtree(
            experiment_path
            + "/"
            + dataset
            + "/feature_selection/multisurf/pickledForPhase4"
        )
        print("MultiSURF Pickle Files Deleted")
    except:
        pass
    # Delete older training and testing CV datasets (does not delete any final versions used for training). Older cv datasets might have been kept to see what they look like prior to preprocessing and feature selection.
    if eval(del_oldCV):
        # Delete CV files generated after preprocessing but before feature selection
        files = glob.glob(experiment_path + "/" + dataset + "/CVDatasets/*CVOnly*")
        for f in files:
            try:
                os.remove(f)
                print("Deleted Intermediary CV-Only Dataset Files")
            except:
                pass
        # Delete CV files generated after CV partitioning but before preprocessing
        files = glob.glob(experiment_path + "/" + dataset + "/CVDatasets/*CVPre*")
        for f in files:
            try:
                os.remove(f)
                print("Deleted Intermediary CV-Pre Dataset Files")
            except:
                pass
