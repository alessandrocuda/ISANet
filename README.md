# ISANet: Neural Network Library for Everyone
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<p align="center">
  <a href="https://github.com/alessandrocuda/ISANet">
    <img src="logo/Logo.png" alt="Logo" width=auto height=auto>
  </a>

  <p align="center">
    <a href="https://alessandrocudazzo.it/ISANet"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/alessandrocuda/ISANet">View Demo</a>
    ·
    <a href="https://github.com/alessandrocuda/ISANet/issues">Report Bug</a>
    ·
    <a href="https://github.com/alessandrocuda/ISANet/issues">Request Feature</a>
  </p>
</p>



ISANet library provides a flexible and modular neural network library. It was entirely developed in Python using Numpy as a package for scientific computation and it is the result of the machine learning course held by Professor Alessio Micheli at [Department of Computer Science](https://www.di.unipi.it/en/) of [University of Pisa](https://www.unipi.it/index.php/english). ISANet is composed of low (Keras-like) and high-level (Scikit-learn-like) APIs divided into modules. The idea is to provide an easy but powerfull implementation of a Neural Network library to allow everyone to understand it from the theory to practice. More importat, the library leave open any kind of future work: extend to [JAX](https://github.com/google/jax), CNN layer or optimizer, and so on. In addition the library provides some datasets and a module for model selection (Grid Search and Cross Validation API).

NOTE: ISANet only support SGD with MSE (Mean Square Error) as LOSS function.

## Details
For more details about the library <a href="https://alessandrocudazzo.it/ISANet"><strong>explore the docs</strong></a>.

## Table of Contents 
- [Usage](#usage)
- [Example](#example)
- [Todo](#todo)
- [Contributing](#contributing)
- [License](#license)

## Usage

An example with the **low level api (keras-like)**:

```python
# ...
from isanet.model import Mlp
from isanet.optimizer import SGD, EarlyStopping
from isanet.datasets.monk import load_monk
import numpy as np

X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

#create the model
model = Mlp()
# Specify the range for the weights and lambda for regularization
# Of course can be different for each layer
kernel_initializer = 0.003 
kernel_regularizer = 0.001

# Add many layers with different number of units
model.add(4, input= 17, kernel_initializer, kernel_regularizer)
model.add(1, kernel_initializer, kernel_regularizer)

es = EarlyStopping(0.00009, 20) # eps_GL and s_UP

#fix which optimizer you want to use in the learning phase
model.setOptimizer(
    SGD(lr = 0.83,          # learning rate
        momentum = 0.9,     # alpha for the momentum
        nesterov = True,    # Specify if you want to use Nesterov
        sigma = None        # sigma for the Acc. Nesterov
    ))

#start the learning phase
model.fit(X_train,
          Y_train, 
          epochs=600, 
          #batch_size=31,
          validation_data = [X_test, Y_test],
          es = es,
          verbose=0) 
            
# after trained the model the prediction operation can be
# perform with the predict method
outputNet = model.predict(X_test)
```

what's next? look to [USE_CASES.md](https://github.com/alessandrocuda/ISANet/blob/master/examples/USE_CASES.md) for more example with the **High-Level API (Scikit-learn-like)** and the **Model Selection API**. Instead, [here](https://github.com/alessandrocuda/ISANet/blob/master/examples/README.md) you can find some example scripts.


## TODO
- [ ] Separate bias from the weight matrix for more clarity.
- [ ] Extend with JAX for GPU support.
- [ ] Add Conjugate Gradient Method.
- [ ] Add Quasi Newton method. 

## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

<!-- CONTACT -->
## Contact

Alessandro Cudazzo - [@alessandrocuda](https://twitter.com/alessandrocuda) - alessandro@cudazzo.com

Giulia Volpi - giuliavolpi25.93@gmail.com

Project Link: [https://github.com/alessandrocuda/ISANet](https://github.com/alessandrocuda/ISANet)


<!-- LICENSE -->
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

This library is free software; you can redistribute it and/or modify it under
the terms of the MIT license.

- **[MIT license](LICENSE)**
- Copyright 2019 ©  <a href="https://alessandrocudazzo.it" target="_blank">Alessandro Cudazzo</a> - <a href="mailto:giuliavolpi25.93@gmail.com">Giulia Volpi</a>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/alessandrocuda/ISANet.svg?style=flat-square
[contributors-url]: https://github.com/alessandrocuda/ISANet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/alessandrocuda/ISANet.svg?style=flat-square
[forks-url]: https://github.com/alessandrocuda/ISANet/network/members
[stars-shield]: https://img.shields.io/github/stars/alessandrocuda/ISANet.svg?style=flat-square
[stars-url]: https://github.com/alessandrocuda/ISANet/stargazers
[issues-shield]: https://img.shields.io/github/issues/alessandrocuda/ISANet.svg?style=flat-square
[issues-url]: https://github.com/alessandrocuda/ISANet/issues
[license-shield]: https://img.shields.io/github/license/alessandrocuda/ISANet.svg?style=flat-square
[license-url]: https://github.com/alessandrocuda/ISANet/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/alessandro-cudazzo-0207b1137/