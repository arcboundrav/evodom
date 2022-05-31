import numpy as np

from evolvedominion.engine.combinatorics import partition
from evolvedominion.engine.engine import (
    shuffle,
    transfer_top_piece,
    draw,
)
from evolvedominion.params import (
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
)
from evolvedominion.engine.pieces import (
    Curse,
    Estate,
    Duchy,
    Province,
    Copper,
    Silver,
    Gold,
)


_N_COPPER = 60
_N_SILVER = 40
_N_GOLD = 30
_N_KINGDOM_CARD = 10
_N_STARTING_COPPER = 7
_N_STARTING_ESTATE = 3

_ESTATE_PILE_INDEX = 1
_PROVINCE_PILE_INDEX = 3
_COPPER_PILE_INDEX = 4

_INITIAL_HAND_SIZE = 5
_EMPTY_PILE_LIMIT = 2
_MAX_N_TURNS = 180


class State:
    """
    Represent the state of a game and the process of arriving at
    initial conditions. Supports accessing Pieces in the Supply
    using arbitrary predicates; and, drawing inferences about
    whether a given change to the Supply would trigger the end of
    the game on a given turn. The game loop antecedent is computed
    by the game_over property.
    """
    __slots__ = (
        "kingdom",
        "n_total_turns_played",
        "province_pile_empty",
        "n_empty_piles",
        "piece_to_pile_size_map",
        "piles",
        "TRASH",
        "players",
        "current_player_index",
        "current_player",
        "n_players",
        "n_action",
        "n_buy",
        "n_coin",
        "merchant_silver_bonus",
        "n_silver_played_this_turn",
        "need_action_phase",
        "need_treasure_phase",
        "need_buy_phase",
    )
    def __init__(self, players, kingdom):
        self.kingdom = kingdom
        self.n_total_turns_played = 0
        self.province_pile_empty = False
        self.n_empty_piles = 0
        self.piece_to_pile_size_map = dict()
        self.piles = list()
        self.TRASH = list()

        self.players = list()
        self.current_player_index = 0
        self.current_player = None
        self.n_players = 0

        self.n_action = 1
        self.n_buy = 1
        self.n_coin = 0
        self.merchant_silver_bonus = 0
        self.n_silver_played_this_turn = 0
        self.need_action_phase = True
        self.need_treasure_phase = True
        self.need_buy_phase = True

        self.accept_players(players)


    def _initialize_empty_piles(self):
        self.piles = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]


    def _solve_pile_sizes(self):
        N_CURSE = (10 * self.n_players) - 10
        N_VICTORY_CARD = 12 if (self.n_players > 2) else 8
        N_ESTATE = N_VICTORY_CARD + (self.n_players * 3)
        self.piece_to_pile_size_map = {
            Curse:N_CURSE,
            Estate:N_ESTATE,
            Duchy:N_VICTORY_CARD,
            Province:N_VICTORY_CARD,
            Copper:_N_COPPER,
            Silver:_N_SILVER,
            Gold:_N_GOLD,
        }
        for card in self.kingdom:
            self.piece_to_pile_size_map[card] = _N_KINGDOM_CARD


    def _fill_piles(self):
        for pile_index, (piece, pile_size) in enumerate(self.piece_to_pile_size_map.items()):
            for i in range(pile_size):
                card = piece()
                card.total_order_index = pile_index
                self.piles[pile_index].append(card)


    def _generate_piles(self):
        self._initialize_empty_piles()
        self._solve_pile_sizes()
        self._fill_piles()


    def _solve_player_opponents(self, player, player_index):
        """ Ensure that opponents are stored in turn order. """
        for j in range(1, self.n_players):
            opponent_index = (player_index + j) % self.n_players
            player.opponents.append(self.players[opponent_index])


    def _provide_starting_deck(self, player):
        COPPER_PILE = self.piles[_COPPER_PILE_INDEX]
        ESTATE_PILE = self.piles[_ESTATE_PILE_INDEX]
        for i in range(_N_STARTING_COPPER):
            transfer_top_piece(destination=player.DECK, source=COPPER_PILE)
            if (i < _N_STARTING_ESTATE):
                transfer_top_piece(destination=player.DECK, source=ESTATE_PILE)
        shuffle(player.DECK)


    def _provide_starting_hand(self, player):
        draw(state=self, actor=player, n=_INITIAL_HAND_SIZE)


    def _setup_player(self, player):
        player.refresh()
        player.state = self
        self._provide_starting_deck(player)
        self._provide_starting_hand(player)


    def _calibrate_players(self):
        """
        Ensure players are ready to start the game and that opponent
        queries are ready for lookup, in turn order.
        """
        for i, player in enumerate(self.players):
            self._setup_player(player)
            self._solve_player_opponents(player, i)


    def _update_player_based_state_variables(self, players):
        """
        Update state variables which are a function of the players and
        randomize the turn order.
        """
        self.players = players
        self.n_players = len(players)
        shuffle(self.players)
        self.current_player_index = self.n_players - 1


    def accept_players(self, players):
        """ Prepare the initial state given a set of players. """
        self._update_player_based_state_variables(players)
        self._generate_piles()
        self._calibrate_players()


    def update_current_player(self):
        self.current_player_index = (self.current_player_index + 1) % self.n_players
        self.current_player = self.players[self.current_player_index]
        self.current_player.increment_turn_count()


    def refresh_turnwise(self):
        """ Reset state variables which don't persist across turns. """
        self.n_action = 1
        self.n_buy = 1
        self.n_coin = 0
        self.merchant_silver_bonus = 0
        self.n_silver_played_this_turn = 0
        self.need_action_phase = True
        self.need_treasure_phase = True
        self.need_buy_phase = True


    def cleanup(self):
        player = self.current_player
        player.DISCARD.extend(player.HAND)
        player.HAND.clear()
        player.DISCARD.extend(player.ASIDE)
        player.ASIDE.clear()
        player.DISCARD.extend(player.PLAY)
        player.PLAY.clear()
        self._provide_starting_hand(player)


    def filter_supply(self, predicate):
        return [pile[-1] for pile in self.piles if (pile and predicate(pile[-1]))]


    def acquisition_ends_game(self, acquisition):
        """
        Determine whether removing the Acquisition's gained piece from
        the Supply will trigger the end of the game.
        """
        gained_piece = acquisition.gained_piece
        source_zone = self.piles[gained_piece.total_order_index]
        if (len(source_zone) == 1):
            if gained_piece.is_province:
                return True
            elif (self.n_empty_piles == _EMPTY_PILE_LIMIT):
                return True
        return False


    @property
    def almost_over(self):
        """ Signal that a single Acquisition could end the game. """
        return ((self.n_empty_piles == _EMPTY_PILE_LIMIT) or (len(self.piles[_PROVINCE_PILE_INDEX]) == 1))


    @property
    def game_over(self):
        return ((self.province_pile_empty or (self.n_empty_piles > _EMPTY_PILE_LIMIT))
                 or (self.n_total_turns_played > _MAX_N_TURNS))
