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
    <img src="Logo/Logo.png" alt="Logo" width="554" height="165">
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



ISANet library provides a flexible and modular neural network library. 

## Project Structure

 - **isanet**: library src code
 
 - **docs**: contains the library documentation. The main file to browse the documentation is docs/index.html

## Details
For more details about the library <a href="https://alessandrocudazzo.it/ISANet"><strong>explore the docs</strong></a>.

## Table of Contents 
- [Usage](#usage)
- [Example](#example)
- [Todo](#todo)
- [Support](#contributing)
- [License](#license)


<!--

We called it IsaNet. We wrote the library entirely in Python using Numpy as a package for scientific computation. All the main operations of a neural network, as the feed-forward and the back-propagation algorithms, are performed by using matrices, and layers of nodes are stored in a list of matrices where columns are the weights of a single node (the bias is the first element). This implementation allowed us to speed up the computation compared to an object-oriented structure (layers, nodes views as object); this was possible thanks to Numpy that can efficiently perform matrix operation by parallelization under the hood. Numpy uses optimized math routines, written in C or Fortran, for linear algebra operation as Blas, OpenBlas or Intel Math Kernel Library (MKL). IsaNet is composed of low and high-level APIs divided into modules.

## Usage
You need to install [swi-prolog](https://www.swi-prolog.org/): a Prolog interpreter 

```bash
# Clone ISA Project
git clone https://github.com/alessandrocuda/ISA.git
cd ISA

# Start swi-prolog
swipl

#Welcome to SWI-Prolog...
?- [start_isa].
```
or

```bash
# Start swi-prolog
swipl -s start_isa.pl
```



## Example
    Hi, I'm ISA your personal assistant for movies and TV show!
    > hi

    Hi, type "help" if you need help
    yes i need help

    Sure, I'm your personal assistant for movies and tv shows!
    For example you can ask me "what movie do you suggest me?" or "i want to watch a tv show"
    > i would like to watch a movie

    Oh, let me think ... maybe "Your Name"?
    > oh thanks

    Your Welcome!
    > 
or if you want to read some other examples: [USE_CASES.md](https://github.com/alessandrocuda/ISA/blob/master/USE_CASES.md)
-->

## TODO
- [ ] Add Conjugate Gradient Method.
- [ ] Add Quasi Newton method. 

## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

or write an email to:
- [alessandro@cudazzo.com](mailto:alessandro@cudazzo.com)
- <giuliavolpi25.93@gmail.com>

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