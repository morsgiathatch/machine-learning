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
Upon selecting Homework 2 problem 2, you can choose any of the five subproblems
of the assignment. Keep in mind that these were written with me just having 
learned python so they have not fully utilized multiprocessing or other more
efficient libraries like numpy. Some of the run-times reach 20 minutes for the
subproblem. Note that this is still significantly quicker than for the students
on Canvas who said they could not get theirs run quicker than 2 hours. 

If you are not running the shell script to reproduce the results in the report, 
you can use the the Bagging Trees, AdaBoost, or RandomForests modules. 
Again you must pre-process the data. See BankData.py in the DecisionTree package
as an example of data construction to use these algorithms. 

## Linear Regression
Upon selecting Homework 2 problem 4, you can choose any of the three subproblems
of the assignment. These are all fairly quick for the small "concrete" dataset. If
are not running the shell script to reproduce the results in the report, 
GradientDescent requires you to first construct the features as a numpy matrix and
the resultant lables as a numpy array. Additionally, this gradient descent is not
passed a gradient yet so is not portable to other uses other than linear regression.


