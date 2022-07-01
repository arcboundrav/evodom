# evolvedominion
A text-based interface for evolving---and, playing against---strategies for Dominion.

This project was created as a proof of concept that minimally sophisticated agents
which rely only on local information could attain competent play through the use
of a genetic algorithm.

![Tests](https://github.com/arcboundrav/evodom/actions/workflows/tests.yml/badge.svg)

## Installation

Before installing it is recommended to create and activate a [virtualenv](https://docs.python.org/3/tutorial/venv.html) using a version of [Python](https://www.python.org/downloads/) >= 3.8.12.

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

## Example

After installation evolvedominion can be run from the command line.

### Evolving Strategies

Strategies need to be evolved prior to being able to play against them.
The following command will run the genetic algorithm for 100 generations
with a population size of 128. The results will be stored using the key
"demo". This key will be used later to play against the best ones.
```
evodom evolve demo --ngen 100 --nstrat 128
```
Note: The number of Strategies per generation must be in the closed
interval [8, 512] and be evenly divisible by 4. The algorithm is capped at 9999
generations per run. Experience shows that a few hundred generations is more
than adequate to evolve competent Strategies.

Note: Passing the -o flag permits the overwriting of past results to reuse
a key. Valid keys are non-empty alphanumeric strings.


### Playing Against Evolved Strategies

The following command will launch a text-based game against the three
strongest Strategies evolved above:
```
evodom play demo
```

### Interface

Type '?' at the prompt to list the available commands.
Exit the game early at any time with CTRL+C.
