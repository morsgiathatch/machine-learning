# Machine Learning

This is a machine learning library developed by Corbin Baldwin for 
CS5350/6350 at the University of Utah.

## To Run
Simply open a shell and change to the directory containing the run.sh file. 
Note that this shell script only works for unix machines. Once in the directory,
simply run ./run.sh and follow the prompts. You will be asked to enter integers 
for the options to choose between the homeworks and the subproblems.

## Decision Tree
To test the decision tree library, you can download all the files and run 
the run.sh script to reproduce results necessary for class (made sure that 
you cd into DecisionTree first). If you just wish
to use the ID3 algorithm, you are responsible for constructing the 
dataset to be read in to Id3. You can use the Metrics class to use any of
the three metrics used for this project. You should additionally use the
provided Node class for the tree structure. After making an instance of Id3,
calling the method id3 with the appropriate arguments will return a 
reference to the root node with which you can traverse the tree however 
you want. For an example, see getTestResult() in BankData.py. The signature
of id3 is:     
```python
id3(examples, attributes, prev_value, labels, current_depth, max_depth, metric):
```
The examples should be the list of training examples. It is advised to 
follow the class declarations of Attribute and Example in BankData.py. 
Attributes is a list containing feature attributes, labels is the set of 
possible feature labels, and metric is a reference to the desired metric
contained in Metrics.py. Upon initialization, prev_value should be set to
None, current_depth should be set to 0, and max_depth should be the desired
maximum tree height. The implementation of id3 allows the user to terminate
the depth of tree using the max_depth argument.

## Ensemble Learning
[//]: # (Upon selecting Homework 2 problem 2, you can choose any of the five subproblems
of the assignment. Keep in mind that these were written with me just having 
learned python so they have not fully utilized multiprocessing or other more
efficient libraries like numpy. Some of the run-times reach 20 minutes for the
subproblems. Note that this is still significantly quicker than for the students
on Canvas who said they could not get theirs run quicker than 2 hours. )

If you are not running the shell script to reproduce the results in the report, 
you can use the the Bagging Trees, AdaBoost, or RandomForests modules. 
Again you must pre-process the data. See BankData.py in the DecisionTree package
as an example of data construction to use these algorithms. 

To use AdaBoost, which currently has a somewhat poor design, create data as in
BankData.py and call 
```python
run_Adaboost(data, test_data, t_value)
```
where t_value is the number of desired trees to create in the boosting algorithm. 
The algorithm returns a python list of the test and train percentages and a list 
containing all the created classifiers. To be precise, it returns a list as
```python
[[test_percentage_float, train_percentage_float], [t_value_amount_of_classifiers]]
```
Feel free to change this as it was not written too well. 

To use BaggingTrees, again create a data structure object in the vein of BankData.py, 
then call
```python
run_bagging_trees(t_value, data_examples, attributes, labels, factor, print_status_bar)
```
where t_value is the size of the desired bag of trees, examples is a python list of lists
of examples containing features (preferably numeric), a list of attributes, a list
of labels, "factor" which is the factor of the data you wish to use to construct the 
trees (i.e., a factor of 3 uses 1/3 of the data training examples), and lastly a 
boolean print_status_bar if you would like the algorithm to print a status bar to 
inform the user (it can be slow for large enough t_value). The data structures passed
in should be as created from the data structure object.
  
Lastly to use RandomForests, call
```python
run_random_forests(t_value, examples, attributes, labels, size, print_status_bar)
```
where t_value is the number of desired trees in the forest, examples, attributes, and 
labels are as above, size is a parameter to create a fixed-cardinality subset of 
the attributes (i.e., if you only want at most 3 random attributes in the subset,
pass in 3), and print_status_bar is a bool whether to give the user a progress update
(which again is useful for large t_value).

[//]: # (If you choose problem 3, this corresponds to the extra credit programming option. Due to the much 
larger dataset, running statistics on 1000 trees took very much too long so I 
reduced to 100 trees. The runtime is still about 30 minutes even with this 
reduction.) 

## Linear Regression
[//]: # (Upon selecting Homework 2 problem 4, you can choose any of the three subproblems
of the assignment. These are all fairly quick for the small "concrete" dataset. If
are not running the shell script to reproduce the results in the report, 
GradientDescent requires you to first construct the features as a numpy matrix and
the resultant lables as a numpy array. Additionally, this gradient descent is not
passed a gradient yet so is not portable to other uses other than linear regression.)

If you are not running the shell script to reproduce the results in the report, 
you can use the the GradientDescent module. If you wish to use the batch gradient
descent algorithm for linear regression, call
```python
run_gradient_descent(features, output, max_iters, constant_step_size, tolerance)
```
where features is a numpy array (not a matrix) of all the features in the train data, 
output is a numpy array of all the labels of the train data, max_iters is a
terminating condition (since we use a fixed step-size, the algorithm is not guaranteed 
to converge), and a tolerance for terminating the algorithm. The algorithm returns 
a python list as the following
```python
[[w_vector], num_iters, [evaluated_costs]]
```
where w_vector is a numpy array of the most updated weight vector, num_iters is the
iteration when the algorithm terminated (in case the user needs the result) and
evaluated_costs is a python list of the cost function evaluated at each iteration.

If you wish to use the stochastic gradient descent, you can run
```python
run_stochastic_grad_descent(features, output, max_iters, constant_step_size, tolerance)
```
where all the inputs are the same as for run_gradient_descent. It returns a python
array conataining the same as for run_gradient_descent.

Lastly, if you wish to have the analytic solution, if the dimension of the column space
is larger than the dimension of the row space (i.e., more rows than columns), then
you can call 
```python
get_analytic_solution(features, output)
```
where features and output are again the same as above. It returns a numpy array of
the analytic vector solution.

##Perceptron 

This is a fairly simple package to implement and to use. If you are grading this 
portion, all the different portions of Homework 3 problem 2 can be accessed by the 
prompts starting from running the script run.sh. Simply enter integers for the choices.
Part d.) is the only program that runs slightly slowly. As explained in the report,
I calculated averages of the average errors to account for errors generated by 
shuffling of the examples. It has a runtime of approximately 1 minute. I've included
a progress bar for your convenience.

For any of algorithms, you should create a dataset in the vein of BankNote.py. It is 
a class with two data members: a python list of the float features as a numpy array
and a python list
of the float labels for each example. In addition, for the perceptron algorithms to
run you must make sure that the example labels are in {-1, +1}. 

For the standard perceptron algorithm, you will call
```python
perceptron(num_epochs, examples, labels, r_step)
```
where num_epochs is the number of iterations to be made through the dataset, examples 
is the numpy array of numpy array features for each example, labels is a numpy 
array of the example labels, and r_step is the desired step size for the algorithm.
The function signatures for the voted and averaged perceptrons are identical except
with the function name.

To calculate the prediction, you can use the get_prediction function with signature
```python
get_prediction(w_vector, x_vector)
```
where both w_vector and x_vector are numpy arrays. The w_vector is simply the 
w_vector returned by both perceptron() and averaged_perceptron() and x_vector is
a numpy array of a single test example. This method does not work for voted_perceptron.
For the voted perceptron, use get_voted_prediction with signature
```python
get_voted_prediction(voted_ret_array, x_vector)
```
where x_vector is the same as get_prediction but voted_ret_array is the array
returned by voted_perceptron. It is a python list [[], []] where the zeroeth sub-list
is a python list of the distinct weight_vectors as numpy arrays and the first 
sub-list is a python list of the counts of the respective vectors. You can pass this
directly into get_voted_prediction() if you just need the prediction or you can use
it however you want.

Both the get_*_prediction() methods return an element of {-1, +1}.