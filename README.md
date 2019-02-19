# Machine Learning

This is a machine learning library developed by Corbin Baldwin for 
CS5350/6350 at the University of Utah.

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
id3 is:     
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