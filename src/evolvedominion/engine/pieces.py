from evolvedominion.params import (
    NONTERMINAL,
    TERMINAL_WITH_DRAW,
    TERMINAL_WITHOUT_DRAW,
)
from evolvedominion.engine.engine import (
    add_card,
    add_action,
    add_buy,
    add_coin,
    add_merchant_silver_bonus,
    add_silver,
    Consequence,
    Effect,
    RemodelChoices,
    CellarChoices,
    WorkshopChoices,
    MilitiaChoices,
    MineChoices,
    ChapelChoices,
    HarbingerChoices,
    VassalChoices,
    BureaucratChoices,
    MoneylenderChoices,
    PoacherChoices,
    LibraryChoices,
    ThroneRoomChoices,
    BanditChoices,
    CouncilRoomChoices,
    SentryChoices,
    WitchChoices,
    ArtisanChoices,
)


class Piece:
    """
    Base class for Dominion pieces.

    total_order_index is an ordinal representing the identity of
    Piece for use in constructing masks for selection methods.

    simple_effects contains the automatic Effects entailed by playing
    the Piece---e.g, drawing 3 cards by playing Smithy.

    decision is a generator of distinct Consequences entailed by
    playing the Piece which must be selected between by the player.

    NOTE
    Profiling indicated that using predicate attributes is more
    efficient for filtering Pieces than using inheritance and
    isinstance() to filter based on typing.
    """
    __slots__ = (
        'cost',
        'points',
        'total_order_index',
        'is_treasure',
        'is_action',
        'is_victory',
        'is_dynamic_victory',
        'is_potential_terminator',
        'is_province',
        'action_class',
        'simple_effects',
        'decision',
    )
    def __init__(self, cost, points=0):
        self.cost = cost
        self.points = points
        self.total_order_index = 0
        self.is_treasure = False
        self.is_action = False
        self.is_victory = False
        self.is_dynamic_victory = False
        self.is_potential_terminator = False
        self.is_province = False
        self.action_class = None
        self.simple_effects = None
        self.decision = None

    def __repr__(self):
        return self.__class__.__name__


class Curse(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=0, points=-1)
        self.total_order_index = 0


class Copper(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=0)
        self.total_order_index = 4
        self.is_treasure = True
        self.simple_effects = Consequence(Effect(add_coin, n=1))


class Gold(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=6)
        self.total_order_index = 6
        self.is_treasure = True
        self.simple_effects = Consequence(Effect(add_coin, n=3))


class Province(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=8, points=6)
        self.total_order_index = 3
        self.is_victory = True
        self.is_province = True


class Smithy(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 12
        self.is_action = True
        self.action_class = TERMINAL_WITH_DRAW
        self.simple_effects = Consequence(Effect(add_card, n=3))


class Silver(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=3)
        self.total_order_index = 5
        self.is_treasure = True
        self.simple_effects = Consequence(Effect(add_silver))


class Village(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=3)
        self.total_order_index = 11
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=1),
                                          Effect(add_action, n=2))


class Estate(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=2, points=1)
        self.total_order_index = 1
        self.is_victory = True


class Remodel(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 13
        self.is_action = True
        self.is_potential_terminator = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = RemodelChoices()


class Duchy(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5, points=3)
        self.total_order_index = 2
        self.is_victory = True


class Merchant(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=3)
        self.total_order_index = 9
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=1),
                                          Effect(add_action, n=1),
                                          Effect(add_merchant_silver_bonus))


class Cellar(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=2)
        self.total_order_index = 7
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_action, n=1))
        self.decision = CellarChoices()


class Moat(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=2)
        self.total_order_index = 8
        self.is_action = True
        self.action_class = TERMINAL_WITH_DRAW
        self.simple_effects = Consequence(Effect(add_card, n=2))


class Workshop(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=3)
        self.total_order_index = 10
        self.is_action = True
        self.is_potential_terminator = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = WorkshopChoices()


class Militia(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 14
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.simple_effects = Consequence(Effect(add_coin, n=2))
        self.decision = MilitiaChoices()


class Market(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 15
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=1),
                                          Effect(add_action, n=1),
                                          Effect(add_buy, n=1),
                                          Effect(add_coin, n=1))


class Mine(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 16
        self.is_action = True
        self.is_potential_terminator = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = MineChoices()


class Chapel(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=2)
        self.total_order_index = 17
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = ChapelChoices()


class Harbinger(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=3)
        self.total_order_index = 18
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=1),
                                          Effect(add_action, n=1))
        self.decision = HarbingerChoices()


class Vassal(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=3)
        self.total_order_index = 19
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.simple_effects = Consequence(Effect(add_coin, n=2))
        self.decision = VassalChoices()


class Bureaucrat(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 20
        self.is_action = True
        self.is_potential_terminator = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = BureaucratChoices()


class Gardens(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 21
        self.is_victory = True
        self.is_dynamic_victory = True

    def solve_points(self, cards):
        return len(cards) // 10


class Moneylender(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 22
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = MoneylenderChoices()


class Poacher(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 23
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=1),
                                          Effect(add_action, n=1),
                                          Effect(add_coin, n=1))
        self.decision = PoacherChoices()


class ThroneRoom(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=4)
        self.total_order_index = 24
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.decision = ThroneRoomChoices()



class Bandit(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 25
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.is_potential_terminator = True
        self.decision = BanditChoices()


class CouncilRoom(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 26
        self.is_action = True
        self.action_class = TERMINAL_WITH_DRAW
        self.simple_effects = Consequence(Effect(add_card, n=4),
                                          Effect(add_buy, n=1))
        self.decision = CouncilRoomChoices()


class Festival(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 27
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_action, n=2),
                                          Effect(add_buy, n=1),
                                          Effect(add_coin, n=2))


class Laboratory(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 28
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=2),
                                          Effect(add_action, n=1))


class Library(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 29
        self.is_action = True
        self.action_class = TERMINAL_WITH_DRAW
        self.decision = LibraryChoices()


class Sentry(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 30
        self.is_action = True
        self.action_class = NONTERMINAL
        self.simple_effects = Consequence(Effect(add_card, n=1),
                                          Effect(add_action, n=1))
        self.decision = SentryChoices()


class Witch(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=5)
        self.total_order_index = 31
        self.is_action = True
        self.action_class = TERMINAL_WITH_DRAW
        self.simple_effects = Consequence(Effect(add_card, n=2))
        self.decision = WitchChoices()


class Artisan(Piece):
    __slots__ = tuple()
    def __init__(self):
        super().__init__(cost=6)
        self.total_order_index = 32
        self.is_action = True
        self.action_class = TERMINAL_WITHOUT_DRAW
        self.is_potential_terminator = True
        self.decision = ArtisanChoices()
