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

Options:
* -g Gamma: Set gamma in the radial basis kernel function (default 1)
* -c Cost: Set the SVM Cost (default 1)
* -w Working_set_size: Size of the Least Squares Problem in every iteration (default 500)
* -t Number_of_Threads: It is the number of threads in the parallel task (default 1)

Example:

    ./PIRWLS-train -g 0.001 -c 1000 -t 4 training_set_file.txt model_file.mod



Test:
_____

To make predictions with the model in a different dataset:

    ./PIRWLS-predict [options] dataset_file model_file output_file

Options:
* -t Number_of_Threads: It is the number of threads in the parallel task (default 1)
* -l Labeled:  (default 0)
    * 1 if the dataset is labeled (shows accuracy)
    * 0 if the dataset is unlabeled

Example:

    ./PIRWLS-predict -t 4 -l 1 dataset_file.txt model_file.mod output_file.txt

Input file format:
=================

The dataset must be provided in LibSVM format, labeled to train the model and labeled or unlabeled for predictions (using the -l option in the PIRWLS-predict command to tell if the file is labeled or unlabeled):

Labeled example:

+1 1:5 7:2 15:6
+1 1:5 7:2 15:6 23:1
-1 2:4 3:2 10:6 11:4

Unlabeled example:

1:5 7:2 15:6
1:5 7:2 15:6 23:1
2:4 3:2 10:6 11:4



