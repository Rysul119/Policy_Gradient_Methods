# Policy Gradient Methods

This repository contains implementation of numerous state-of-the-art policy gradient methods to perform Reinforcement Learning using [tensorflow 2.x](https://www.tensorflow.org/).

### Implemented Algorithms
 - REINFORCE [paper link]( http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
 - Actor Critic
 - Advantage Actor Critic [paper link](https://arxiv.org/pdf/1602.01783.pdf)
 - Trust Region Policy Optimization [paper link](https://arxiv.org/pdf/1502.05477.pdf)
 - Proximal Policy Optimization [paper link](https://arxiv.org/pdf/1707.06347.pdf)

You will need to install [python](https://www.python.org), [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org), [gym](https://gym.openai.com/), and [tensorflow 2.x](https://www.tensorflow.org/). If you have anaconda installed, run the following:
```bash
conda create -n envName python numpy pandas matplotlib 
```
This will create a conda environment with python, numpy, pandas, and matplotlib installed in it. Run `conda activate envName` to activate or `conda deactivate` to deactivate the environment. Then run the following commands to install [gym](https://gym.openai.com/) and [tensorflow 2.x](https://www.tensorflow.org/) after activating the `conda` environment.
```bash
pip install gym tensorflow
```