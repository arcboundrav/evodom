import numpy as np

from itertools import chain, filterfalse
from evolvedominion.engine.engine import TreasureChoices


class Agent:
    """
    Base class for participants in a game of Dominion.
    """
    __slots__ = (
        "pid",
        "state",
        "score",
        "n_turns_played",
        "opponents",
        "HAND",
        "DISCARD",
        "ASIDE",
        "PLAY",
        "DECK",
        "zones",
        "include",
        "exclude",
    )
    def __init__(self, pid):
        self.pid = pid
        self.state = None
        self.score = 0
        self.n_turns_played = 0
        self.opponents = []
        self.HAND = []
        self.DECK = []
        self.DISCARD = []
        self.ASIDE = []
        self.PLAY = []
        self.zones = [
            self.HAND,
            self.DECK,
            self.DISCARD,
            self.ASIDE,
            self.PLAY,
        ]
        self.include = []
        self.exclude = []

    def refresh(self):
        """ Calibrate attributes. """
        self.state = None
        self.score = 0
        self.n_turns_played = 0
        self.opponents.clear()
        for zone in self.zones:
            zone.clear()

    def increment_turn_count(self):
        self.n_turns_played = self.n_turns_played + 1

    @property
    def collection(self):
        """
        Collects all of the Pieces in an Actor's zones, adding any in
        self.include and removing any in self.exclude.

        By default self.include and self.exclude are empty. They are
        used to produce hypothetical post-Acquisition collections to
        support looking ahead at victory point changes without altering
        any zones or Supply piles.
        """
        return list(filterfalse(lambda piece: (piece in self.exclude),
                                chain(*self.zones, self.include)))


    @property
    def victory_points(self):
        cards = self.collection
        return sum(piece.solve_points(cards) if piece.is_dynamic_victory else piece.points for piece in cards)


    def _automatic(self, choices, decision):
        # Case: Automatically make forced choices.
        if (len(choices) == 1):
            return choices[0]

        # Case: Play all of the treasures available during the Treasure Phase.
        elif isinstance(decision, TreasureChoices):
            # NOTE #
            # choices[0] will end the treasure phase without playing
            # any treasures.
            # choices[1] will be the only other option, to play all of
            # the treasures.
            # choices[1] is guaranteed to exist by the single choice
            # short-circuit above.
            return choices[1]
        return None


    def select(self, choices, decision):
        raise NotImplementedError("Actor subclasses must define their own select method.")


    def peek(self, cards):
        """ Hook for displaying drawn cards to human players. """
        pass


    def __lt__(self, other):
        """
        Comparison defined according to the win condition of Dominion.
        Allows for sorting players without overriding __eq__.
        (not(Player A < Player B) and not(Player B < Player A)) implies
        Player A and Player B are tied.
        ((Player A < Player B) and not(Player B < Player A)) implies
        Player B is beating Player A.
        """
        if isinstance(other, Actor):
            vp, ntp = self.victory_points, self.n_turns_played
            other_vp, other_ntp = other.victory_points, other.n_turns_played
            if (vp < other_vp):
                return True
            elif ((vp == other_vp) and (ntp > other_ntp)):
                return True
        return False

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.pid)
