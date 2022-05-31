from evolvedominion.params import (
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
)
from evolvedominion.engine.engine import (
    resolve,
    ActionChoices,
    TreasureChoices,
    BuyChoices,
)
from evolvedominion.engine.state import State
from evolvedominion.engine.pieces import (
    Cellar,
    Moat,
    Merchant,
    Workshop,
    Village,
    Smithy,
    Remodel,
    Militia,
    Market,
    Mine,
    Chapel,
    Harbinger,
    Vassal,
    Bureaucrat,
    Gardens,
    Moneylender,
    Poacher,
    ThroneRoom,
    Bandit,
    CouncilRoom,
    Festival,
    Laboratory,
    Library,
    Sentry,
    Witch,
    Artisan,
)

_DEFAULT_KINGDOM = [
    Cellar,
    Moat,
    Merchant,
    Workshop,
    Village,
    Smithy,
    Remodel,
    Militia,
    Market,
    Mine,
]

_ANOTHER_KINGDOM = [
    Chapel,
    Harbinger,
    Vassal,
    Bureaucrat,
    Gardens,
    Moneylender,
    Poacher,
    ThroneRoom,
    Bandit,
    CouncilRoom,
]

_ANOTHER_KINGDOM2 = [
    Poacher,
    ThroneRoom,
    Bandit,
    CouncilRoom,
    Festival,
    Laboratory,
    Library,
    Sentry,
    Witch,
    Artisan,
]


class Session:
    """
    Given a set of players, initialize a game of Dominion and control
    their progression through each phase and turn. Award them on the
    basis of their performance.

    Session instances are re-used by Tournaments and are copied and
    passed between processes. Hence the state attribute's persistence
    is intentionally tied to the lifetime of each call to the play
    method.
    """
    __slots__ = "players", "decisions", "state", "kingdom"
    def __init__(self, kingdom=_DEFAULT_KINGDOM):
        self.kingdom = kingdom
        self.players = list()
        self.decisions = {
            ACTION_PHASE:ActionChoices(),
            TREASURE_PHASE:TreasureChoices(),
            BUY_PHASE:BuyChoices()
        }
        self.state = None

    def accept_players(self, players):
        self.players = list(players)

    def order_players(self):
        self.players.sort(reverse=True)

    def award_players(self):
        """
        # Players   1st     2nd     3rd     4th
        2           1       0       ---     ---
        3           2       1       0       ---
        4           3       2       1       0
        """
        self.order_players()
        n_players = self.state.n_players
        for i in range(n_players):
            self.players[i].score = n_players - 1 - i

    def start_turn(self):
        self.state.n_total_turns_played += 1
        self.state.update_current_player()
        self.state.refresh_turnwise()

    def action_phase(self):
        while self.state.need_action_phase:
            resolve(state=self.state,
                    actor=self.state.current_player,
                    decision=self.decisions[ACTION_PHASE])

    def treasure_phase(self):
        resolve(state=self.state,
                actor=self.state.current_player,
                decision=self.decisions[TREASURE_PHASE])

    def buy_phase(self):
        while self.state.need_buy_phase:
            resolve(state=self.state,
                    actor=self.state.current_player,
                    decision=self.decisions[BUY_PHASE])

    def end_turn(self):
        self.state.cleanup()

    def initialize_state(self):
        self.state = State(players=self.players,
                           kingdom=self.kingdom)

    def play(self):
        self.initialize_state()
        while not(self.state.game_over):
            self.start_turn()
            self.action_phase()
            self.treasure_phase()
            self.buy_phase()
            self.end_turn()
        self.award_players()
        self.state = None
