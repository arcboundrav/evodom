import click

from evolvedominion.utils import DATA_MANAGER
from evolvedominion.params import (
    GROUPSIZE,
    MIN_NSTRAT,
    MAX_NSTRAT,
    DEFAULT_NSTRAT,
    DEFAULT_NSTRAT_HELP,
    INVALID_NSTRAT,
    MIN_NGEN,
    MAX_NGEN,
    DEFAULT_NGEN,
    DEFAULT_NGEN_HELP,
    INVALID_NGEN,
    INVALID_SIMNAME,
    INVALID_OVERWRITE,
    DEFAULT_OVERWRITE_HELP,
)


def _is_valid_simname(simname):
    return simname.isalnum()


def _is_valid_nstrat(nstrat):
    return ((MIN_NSTRAT <= nstrat <= MAX_NSTRAT) and not(nstrat % GROUPSIZE))


def _is_valid_ngen(ngen):
    return (MIN_NGEN <= ngen <= MAX_NGEN)


def validate_evolve_args(overwrite, ngen, nstrat, simname):
    if not(_is_valid_simname(simname)):
        print(INVALID_SIMNAME)
        return False
    elif not(_is_valid_ngen(ngen)):
        print(INVALID_NGEN)
        return False
    elif not(_is_valid_nstrat(nstrat)):
        print(INVALID_NSTRAT)
        return False
    elif (not(overwrite) and DATA_MANAGER.requires_overwrite(simname)):
        print(INVALID_OVERWRITE.format(simname))
        return False
    return True


def validate_play_args(simname):
    if not(_is_valid_simname(simname)):
        print(INVALID_SIMNAME)
        return False
    return True


@click.group()
def main():
    pass


@main.command()
@click.option("-o",
              is_flag=True,
              help=DEFAULT_OVERWRITE_HELP)
@click.option("--ngen",
              default=DEFAULT_NGEN,
              help=DEFAULT_NGEN_HELP)
@click.option("--nstrat",
              default=DEFAULT_NSTRAT,
              help=DEFAULT_NSTRAT_HELP)
@click.argument("simname")
def evolve(o, ngen, nstrat, simname):
    """
    Evolve strategies to play against.\n
    simname\tKeyword for reading and writing strategy data.
    """
    if validate_evolve_args(o, ngen, nstrat, simname):
        from evolvedominion.algorithm.evolve import Simulation
        click.echo("Start of simulation {}.".format(simname))
        try:
            simulation = Simulation(simname=simname, N=nstrat)
            simulation.evolve(n_generation=ngen)
        except KeyboardInterrupt:
            click.echo("Simulation aborted. Unable to save data.")
        except Exception as e:
            click.echo("Unexpected error: {}".format(e))
        else:
            click.echo("Attempting to save simulation {} results...".format(simname))
            DATA_MANAGER.save(simulation)


@main.command()
@click.argument("simname")
def play(simname):
    """
    Load and play against the strongest strategies evolved under the
    keyword SIMNAME.
    """
    if validate_play_args(simname):
        from evolvedominion.play import play_game
        play_game(simname)
