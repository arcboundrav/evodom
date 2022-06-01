# evolvedominion
A text-based interface for evolving---and, playing against---strategies for Dominion.

This project was created as a proof of concept that minimally sophisticated agents
which rely only on local information could attain competent play through the use
of a genetic algorithm.

![Tests](https://github.com/arcboundrav/evodom/actions/workflows/tests.yml/badge.svg)

## Installation

Create a [virtualenv](https://docs.python.org/3/tutorial/venv.html) with a version of [Python](https://www.python.org/downloads/) >= 3.8.12.

### Install from PyPI using `pip`
```
python -m pip install -U pip
python -m pip install -U evolvedominion
```

### Install from Git
Clone the repository using either
```
git clone https://github.com/evolvedominion/evolvedominion.git
```
or
```
git clone git@github.com:evolvedominion/evolvedominion.git
```
Navigate to the top level of the package and install using `pip`
```
cd evolvedominion
python -m pip install .
```
Note: The tests will require additional dependencies.
```
python -m pip install -r requirements_dev.txt
```
Updating evolvedominion to the latest release can be done
by navigating to the repository and using `git pull`
