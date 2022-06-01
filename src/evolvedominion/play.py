from evolvedominion.params import GROUPSIZE
from evolvedominion.utils import DATA_MANAGER
from evolvedominion.agents.player import Player, EchoPlayer
from evolvedominion.agents.strategy import (
    Strategy,
    RandomStrategy,
    EchoStrategy,
    EchoRandomStrategy,
)
from evolvedominion.display.echo import EchoSession
from evolvedominion.display.text import newline, summarize_session


def _load_winning_phenotypes(simname):
    simulation_data = DATA_MANAGER.load(simname)
    if ((simulation_data is None) or not(simulation_data)):
        winners = None
        print("Failed to extract strategies from simulation file: {}.".format(simname))
    else:
        winners = simulation_data['winners']
    return winners


def _create_session(winning_phenotypes):
    phenotypes = []
    n_to_play_against = GROUPSIZE - 1
    n_winning_phenotypes = len(winning_phenotypes)
    i = n_winning_phenotypes - 1
    while (len(phenotypes) < n_to_play_against):
        phenotypes.append(winning_phenotypes[i])
        if i:
            i = i - 1
    players = [EchoPlayer(pid=0)]
    players.extend(EchoStrategy(pid=i, phenotype=phenotypes[i]) for i in range(n_to_play_against))
    session = EchoSession()
    session.accept_players(players)
    return session


def play_game(simname):
    winners = _load_winning_phenotypes(simname)
    if (winners is not None):
        session = _create_session(winners)
        try:
            session.play()
        except KeyboardInterrupt:
            # NOTE #
            # No guarantee that players are sorted according to their
            # performance when the session ends unexpectedly.
            session.order_players()
            newline()
            summarize_session(session)
        except Exception as e:
            print("Unexpected error: {}".format(e))
        else:
            summarize_session(session)
