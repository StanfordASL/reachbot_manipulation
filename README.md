![Reachbot](images/banner.jpg)

# Manipulation with Reachbot

Code for *Task-Driven Manipulation with Reconfigurable Parallel Robots* - Daniel Morton, Mark Cutkosky, Marco Pavone

## Getting started:

## Virtual Environment

A virtual environment is optional, but highly recommended. I prefer `pyenv` to `conda` - for `pyenv` installation instructions, see [here](docs/pyenv.md).

```
# pyenv install 3.10.8 if not already installed
pyenv virtualenv 3.10.8 reachbot
pyenv shell reachbot
```

## Clone the repo

```
cd $HOME
git clone https://github.com/StanfordASL/reachbot_manipulation
```

## Install dependencies

```
cd $HOME/reachbot_manipulation
pip install -e .
```

### Additional dependencies:

To install MOSEK, in addition to the pip-install, you will also need a license. Academic licenses are easy to obtain [here](https://www.mosek.com/products/academic-licenses/)


## Usage

The main methods introduced in the paper can be found in the `optimization` folder, and demos can be found in `scripts`
