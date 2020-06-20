# Ceuron
Project to create a Neural Networks Libary

# Current Features
The project can currently run a feed forward neural network. This is a network with only one layer therefore without a hidden layer.

# Usage
```python
import Ceuron

Ceuron.buildFFNN("ReLu", 20, Data, Targets) # ReLu is the transfer function, 20 the number of epochs
```

# Features to come
- Currently working on a Multi Layer Feed Forward Neural Network build, this will be called in a similar manner to the FFNN
- Competitive learning neural network, this will be unsupervised (no targets)
- Convilutional neural network

# Pipeline Goals
- GUI based interface that will allow for the drag and drop of perceptrons to allow the user to build the network themselves
