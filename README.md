<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                            BANNER & SHIELD                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


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

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                               MAIN TITLE                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# MicroNet

Pytorch implementation of the [MicroNet Challenge](https://micronet-challenge.github.io/) based on [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                          TABLE OF CONTENTS                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Table of contents

- [To Do](#to-do-for-next-release)
     - [New Features](#new-features)
     - [Bugfixes](#bugfixes)
- [Last Commit Changes Log](#last-commit-changes-log)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                  TO DO                                             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# To Do for Next Release
[(Back to top)](#table-of-contents)

## New features:

| Features                                                 |      Status      |      Type    |
|----------------------------------------------------------|:----------------:|:------------:|
| Fix validation accuracy computation issue                |  DONE            |   Bugfix     |
| Add current learning rate in terminal                    |  TO DO           |   Bugfix     |
| Add test method in model & check best model              |  DONE            |   Feature    |
| Add loss and acc to Tensorboard                          |  DONE            |   Feature    |
| Terminal size and cursor issue                           |  DONE            |   Bugfix     |
| Add best train acc and best val acc in terminal          |  DONE            |   Feature    |
| No more fastai dependancy                                |  DONE            |   Feature    |


- tensorboard logs: must define a tensorboard logger object (callback) and add scalar to it


## Bugfixes:

- learning rate logging:
     the tensorboard logger is correct.
     the lr displayed in terminal is false: it's constant at the initial lr.

- accuracy computation issue:
     * solved by calculating our own accuracy.
     * is calculated on CPU and not GPU: need to be changed in future releases.

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              CHANGES LOG                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# Last Commit Changes Log

- improve README
- improve terminal display:
     - table showing training logs
     - make verbose a Lightning callback (see utils/verbose.py) instead of a decorator:
       this callback is based on a State class that could be use somewhere else.
- remove fastai dependancy:
     code loaded from fastai is now in pytorch
- test in model and main:
     for now it's only a duplicate of a validation routine on one epoch.


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              INSTALLATION                                          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

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


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                 USAGE                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

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

If you want to debug but the fast_dev_run option doesn't suit (for instance if you wanna check what's happening between
two epochs) you can run:

- ```--limit_train_batches i --limit_val_batches j --max_epochs k```

i,j,k are of course three integers of your choice.



<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                LICENSE                                             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# License
[(Back to top)](#table-of-contents)

Feel free to use, modify, and share this code.
Consider citing us if you feel like it.

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)





