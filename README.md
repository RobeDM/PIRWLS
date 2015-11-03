# PIRWLS

Parallel Iterative Re-Weighted Least Squares: A Parallel Support Vector Machine (SVM) solver based on the IRWLS algorithm

Instalation:
============

This software is implemented in C and requires the following libraries:

 - [OpenMP] (http://openmp.org/wp/)
 - [Scipy] (http://www.scipy.org/)
 - [SciKit Learn](http://scikit-learn.org/)


Settings:
=========

This software contains two setting files:

'Variables.py' contains the environment variables that the software needs:

 * There is an environment variable for the absolute path of every file provided in the challenge.
 * There is an environment variable for the python wrapper folder of XGBoost.
 * There is an environment variable for the path to save/load the model.

'VariablesTST.py' contains two environment variables:

 * There is an environment variable for the absolute path of the test file.
 * There is an environment variable for the absolute path of the file with the results to submit to kaggle.

Edit 'Variables.py' and 'VariablesTST.py' and use the values of you files and the path that contains the wrapper of your installation of XGBoost.

The procedure 'train.py' makes use of 'Variable.py'
The procedure 'test.py' makes use of 'Variable.py' and 'VariableTST.py' (it uses Variable.py because the model has very big data structures that is more practical to load again from the training files than saving and loading again, if you want to evaluate your model in a different test file than the one provided for the challenge uses the variable "testfile" in 'Variables.py' for the original test file provided by Drawbridge and the variable "predictFile" in 'VariablesTST.py' for the file that you wish to evaluate).

Requirements:
=============

The experiments were executed on a HP DL160 G6 server with 48 GBytes and 2 Intel Xeon X5675 processors (each one has 6 cores with hyperthreading technology).
The operating system was linux gentoo

(32 GB of RAM should be enough).

Running the code:
=================

Training:
________

To train the algorithm and create the model:

    ./PIRWLS-train [options] training_set_file model_file

Test:
_____

To make predictions with the model in a different dataset:

    ./PIRWLS-predict [options] dataset_file model_file output_file



