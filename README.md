# Machine Learning

This is an introductory machine learning library developed. 
It offers the following machine learning algorithms:
* Decision Trees
* Adaboost
* Bagging Trees
* Random Forest
* Linear Regression
* Perceptron
* Kernel Perceptron
* SVM
* Kernel SVM
* Three Layered Neural Network
* MAP Estimation
* ML Estimation

For information on how to use these, see the Documentation section below. Most of the 
algorithms operate similarly to those developed for `scipy` among other libraries 
with the exception being SVM and Kernel SVM. First, 
an object of the specific algorithm must be instantiated. After this, `fit` must be 
called to fit the algorithm to the data. Lastly, `predict` is used to return the
prediction of a feature. The assumption is made that the user is familiar with the above
algorithms.
## Data
This folder contains the data necessary to run the tests. There are five datasets that
the user must download and move into the corresponding folders in order to run the tests.
The url's are as follows:
* Bank data should be downloaded from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
  and stored in `Data/bank`
* Bank Note data does not have an available url. Contact me if you wish for a copy of this dataset.
  It must be stored in `Data/bank_note`
* Car data should be downloaded from https://archive.ics.uci.edu/ml/datasets/car+evaluation
  and stored in `Data/car`
* Concrete data should be downloaded from https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test 
  and stored in `Data/concrete`
* Credit data should be downloaded from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
  and stored in `Data/credit`

The details about each dataset are given in each of the links. **Note:** Some of the datasets
may not work correctly if they have changed or require sanitation. I was initially given
fairly sanitized datasets in .csv format. By *sanitized* I mean that incomplete information
was removed so that only well-formed features are present (i.e., each feature must 
have the same dimension).
## Tests
This folder does not contain tests in the usual sense of unit testing. Rather it 
is a collection of tests that verify that the algorithm in question is in fact
learning. To run the tests, if you have downloaded the appropriate datasets, simply run
```bash
bash ./run.sh
```
and follow the prompts to run the tests explained below.
1. Decision Tree testing has two parts
    1.  Non-numeric ID3 algorithm allowing a user-desired metric (see ID3 above) as
    well as a user-desired tree depth. For statistical purposes, a prompt asks if 
    you would like to run some simple statistics over all implemented metrics and
    tree depths (up to maximum tree depth). This test uses the `car` dataset.
    1. Numeric ID3 algorithm offers the same features as the non-numeric ID3 in addition 
    to allowing 'unknown' attribute values to be considered attributes or to be filled
    in with some other attribute value. In our case, we replace the 'unknown' attribute
    value with the most common attribute value. As before, simply follow the prompts. This 
    test uses the `bank` dataset.
1. Ensemble Learning testing has six parts
    1. Runs 1000 epochs of Adaboost and reports how the training and 
    test errors change over the epochs. This test uses the `bank` dataset.
    1. Runs 1000 epochs of Bagged Trees and reports how the training and 
    test errors change over the epochs. This test uses the `bank` dataset.
    1. Runs a statistical cross comparison reporting the bias and variance of the training 
    and testing errors for the Adaboost and Bagged Trees algorithms. This test uses the `bank` dataset.
    1. Runs 1000 epochs of Random Forest and reports how the training and test errors
    change over the epochs for various attribute sizes (2, 4, or 6). This test 
    uses the `bank` dataset. 
    1. Runs a statistical cross comparison reporting the bias and variance of the training
    and testing errors for the Random Forest and Bagged Trees algorithms. This test
    uses hte `bank` dataset.
    1. Runs a cross comparison of Adaboost, Bagged Trees, and Random Forest algorithms.
    Reports the training and testing errors over 1000 epochs for the `credit` dataset.
1. Linear Regression testing has two parts
    1. Run the standard gradient descent algorithm using function 
    f(x) = 1/2 ||X * w - y|| and graphically reports the convergence to a solution.
    This test uses the `concrete` dataset.
    1. Run the stochastic gradient descent algorithm using function 
    f(x) = 1/2 ||X * w - y|| and graphically reports the convergence to a solution.
    This test uses the `concrete` dataset.
    1. Analytically solve the optimization problem for the objective function above and
    report the weight vector and optimized cost. This test uses the `concrete` dataset.

1. Perceptron testing has five parts
    1. Run the standard Perceptron algorithm for 10 epochs and report the weight vector,
    training, and testing errors. This test uses the `bank_note` dataset.
    1. Run the Voted Perceptron algorithm for 10 epochs and report the weight vector, 
    training, and testing errors as well as the weight vectors whose correct prediction
    throughout the algorithm did not update the weight vector. Also reported is the number
    of features for which each weight vector correctly predicted in a row. 
    This test uses the `bank_note` dataset.
    1. Run the Average Perceptron algorithm for 10 epochs and report the weight vector,
    training, and testing errors. This test uses the `bank_note` dataset.
    1. Run a statistical cross comparison of the above three algorithms and report
    the average training and testing errors for each. This test uses the `bank_note` 
    dataset.
    1. Run the Kernel Perceptron algorithm in the dual domain of the optimization problem
    for 10 epochs and report the weight vector, training, testing errors.This test uses 
    the `bank_note` dataset.
1. Support Vector Machine (SVM) testing has six parts divided into two
    1. Test SVM in the primal domain using various C hyperparameter values
    in between 0 and 1. See the code for specific C values. Uses stochastic
    sub-gradient descent.
        1. Run SVM with learning rate g/(1+(g/d)t) and report the training and
        testing errors. This test uses the `bank_note` dataset.
        1. Run SVM with learning rate g/(1+t) and report the training and 
        testing errors. This test uses the `bank_note` dataset.
        1. Run a cross comparison of the above two learning rates and report
        training and testing errors. This test uses the `bank_note` dataset.
    1. Test SVM in the dual domain using various  C hyperparameter values
    in between 0 and 1. See the code for specific C values. Uses the 
    `minimize` method as found in `scipy.optimize`.
        1. Run SVM in the dual domain and report the training and testing 
        errors. This test uses the `bank_note` dataset.
        1. Run non-linear Kernel SVM in the dual domain and report the 
        training and testing errors. This test uses the `bank_note` dataset.
        1. Run non-linear Kernel SVM in the dual domain and report the
        number of support vectors. This test uses the `bank_note` dataset.
1. Logistic Regression testing has two parts and uses stochastic gradient 
descent
    1. Run MAP Estimation using various values for variance and report the
    training and testing errors.This test uses the `bank_note` dataset.
    1. Run ML Estimation and report the training and testing errors. This 
    test uses the `bank_note` dataset.
1. Neural Network testing using a feed-forward three-layered neural network
    and back-propogation. All layers have the same fixed number of units. It
    has two parts
    1. Run the Neural Network using weights initialized from a standard
    normal distribution and report the training and testing errors. This 
    test uses the `bank_note` dataset.
    1. Run the Neural Network using weights set to the zero tensor and 
    report training and testing errors. This test uses the `bank_note` dataset.
    
## Documentation
Included in the folder `documentation/_build/html` are html pages for the Machine Learning
API. They were developed by the static site generator Sphinx located at https://www.sphinx-doc.org/en/master/.
If you wish to view the API documentation, change to the directory above and open `index.html` 
in a browser. To generate the documentation, run the command
```bash
bash ./gen_docs.sh
```
