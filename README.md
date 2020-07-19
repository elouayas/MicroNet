![](./micronet.png)

<p align="center">
    <!-- Last Master Commit-->
    <img src="https://img.shields.io/github/last-commit/the-dharma-bum/MicroNet?label=last%20master%20commit&style=flat-square"
         alt="GitHub last commit">
        <!-- Last Commit-->
    <img src="https://img.shields.io/github/last-commit/the-dharma-bum/MicroNet/improve_logging?style=flat-square"
         alt="GitHub last commit">
    <!-- Commit Status -->
    <img src="https://img.shields.io/github/commit-status/the-dharma-bum/MicroNet/improve_logging/0c8c2d6e5363b479344983c564c6dcc27834390a?style=flat-square"
         alt="GitHub commit status">
    <br>
    <!-- Issues -->
    <a href="https://github.com/the-dharma-bum/MicroNet/issues">
    <img src="https://img.shields.io/github/issues/the-dharma-bum/MicroNet?style=flat-square"
         alt="GitHub issues">
    <!-- Pull Requests -->
    <a href="https://github.com/the-dharma-bum/MicroNet/pulls">
    <img src="https://img.shields.io/github/issues-pr/the-dharma-bum/MicroNet?color=blue&style=flat-square"
         alt="GitHub pull requests"></a>
    <br>
    <!-- Licence -->
    <img alt="GitHub" src="https://img.shields.io/github/license/navendu-pottekkat/nsfw-filter?style=flat-square&color=yellow">
</p>

# MicroNet

Pytorch implementation of the [MicroNet Challenge](https://micronet-challenge.github.io/) based on [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).



# Table of contents

- [To Do](#to-do-for-next-release)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


# To Do for Next Release
[(Back to top)](#table-of-contents)



## New features:

- clearer terminal display during training:
    - add best train acc and best val acc
    - add current learning rate

## Bugfix:

- accuracy and loss display when training on multiple gpus

# Installation
[(Back to top)](#table-of-contents)

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/the-dharma-bum/MicroNet```

Note that this projet requires fastai and pytorch lightning. 

To ensure everything run ok, you could try:

```apt install gcc git pip```

```pip install fastai```

```pip install pytorch-lightning```



# Usage
[(Back to top)](#table-of-contents)

You can modify any hyper parameters in the config.py file. 
Alternatively, you can declare dataclasses (refer to config.py to see how they should be instanciated) anywhere in your code, then instanciate a model object using those dataclasses, and finally give them to a trainer object. 

Once you're ready, run:

```python main.py ```

This command supports many arguments, type 

```python main.py -h ```

to see them all, or refer to the pytorch-lightning documentation.

Most useful ones:

- ```--gpus n``` : runs the training on n gpus
- ```--distributed_backend ddp``` : use DistributedDataParallel as backend to train across multiple gpus.
- ```--fast_dev_run True``` : runs one training loop, that is one validation step, one test step, one training step on a single data batch. Used to debug efficiently. 


# License
[(Back to top)](#table-of-contents)

Feel free to use, modify, and share this code.
Consider citing us if you feel like it.

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)





