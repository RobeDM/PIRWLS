# PIRWLS

Parallel Iterative Re-Weighted Least Squares: A Parallel Support Vector Machine (SVM) solver based on the IRWLS algorithm

Requirements:
=============

This software is implemented in C and requires the following libraries:

 - [OpenMP] (http://openmp.org/wp/) To parallelize the software
 - A Linear Algebra Package that implements the BLAS and Lapack standard routines, this software has been tested with these two libraries (you need just one of them):
     - [ATLAS] (http://math-atlas.sourceforge.net/)
     - [MKL](https://software.intel.com/en-us/intel-mkl)


Installation Instructions:
=========

External libraries:
________________

An example to install in ubuntu all the libraries that we need is the following:

 * OPENMP is currently included with the gcc compiler, if gcc is not installed, use the following command line :
 
    sudo apt-get install build-essential

 - ATLAS


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

~~~~
+1 1:5 7:2 15:6
+1 1:5 7:2 15:6 23:1
-1 2:4 3:2 10:6 11:4
~~~~

Unlabeled example:

~~~~
1:5 7:2 15:6
1:5 7:2 15:6 23:1
2:4 3:2 10:6 11:4
~~~~


