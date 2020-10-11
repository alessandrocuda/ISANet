# ISANet examples directory

[USE_CASES.md](USE_CASES.md)
Some use cases for high/low level APIs and model selection API.


[iris_mlp.py](sgd/iris_mlp.py)
Trains a simple multi-layer perceptron with a single hidden layer on the IRIS dataset with the keras-like APIs.

[iris_mlp_k_fold.py](model_selection/iris_mlp.py)
K-fold Cross validation with a simple multi-layer perceptron with a single hidden layer on the IRIS dataset with the sklearn-like APIs.

Some test with the MONK Datasets and shallow nets divided by optimizer:

### SGD
- [monk1_test.py](sgd/monk1_test.py)
- [monk2_test.py](sgd/monk1_test.py)
- [monk3_test.py](sgd/monk1_test.py)

### NCG
- [monk1_test.py](ncg/monk1_test.py)
- [monk2_test.py](ncg/monk1_test.py)
- [monk3_test.py](ncg/monk1_test.py)

### L-BFGS
- [monk1_test.py](lbfgs/monk1_test.py)
- [monk2_test.py](lbfgs/monk1_test.py)
- [monk3_test.py](lbfgs/monk1_test.py)

Some test with the MONK Datasets and deep nets:
- [monk1_deep_test.py](deep/monk1_deep_test.py)
- [monk2_deep_test.py](deep/monk2_deep_test.py)




