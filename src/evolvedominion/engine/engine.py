import numpy as np

from functools import partial

from evolvedominion.params import (
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
)
from evolvedominion.engine.combinatorics import (
    partition,
    classify,
    get_pieces,
    get_piece_combinations,
)


def shuffle(iterable):
    np.random.shuffle(iterable)


def transfer_piece(piece, destination, source):
    destination.append(source.pop(source.index(piece)))


def transfer_top_piece(destination, source):
    destination.append(source.pop())


def enact(state, consequence):
    """ Alter the game accordingly. """
    for effect in consequence:
        effect(state)


def expand_choices(state, actor, decision):
    return decision.generate_choices(state, actor)


def request_input(state, actor, decision, decider):
    choices = expand_choices(state, actor, decision)
    return decider.select(choices, decision)


def resolve(state, actor, decision, decider=None):
    """
    Generate the distinct versions of the instructions actor
    should follow given Decision, then have decider select
    which version to follow.
    Note: By default, decider is actor.
    """
    decider = actor if (decider is None) else decider
    consequence = request_input(state, actor=actor, decision=decision, decider=decider)
    enact(state, consequence)


def resolve_effects(state, actor, piece):
    if (piece.simple_effects is not None):
        enact(state, piece.simple_effects)
    if (piece.decision is not None):
        resolve(state, actor, decision=piece.decision)


def add_card(state, n):
    draw(state, actor=state.current_player, n=n)


def add_action(state, n):
    state.n_action = state.n_action + n


def add_buy(state, n):
    state.n_buy = state.n_buy + n


def add_coin(state, n):
    state.n_coin = state.n_coin + n


def add_merchant_silver_bonus(state):
    """
    Support the boost to Silver's value for each Merchant played
    on a Turn.
    """
    state.merchant_silver_bonus = state.merchant_silver_bonus + 1


def add_silver(state):
    add_coin(state, n=2)
    state.n_silver_played_this_turn = state.n_silver_played_this_turn + 1
    if (state.n_silver_played_this_turn == 1):
        add_coin(state, n=state.merchant_silver_bonus)


def end_phase(state, actor, phase):
    if (phase == ACTION_PHASE):
        state.need_action_phase = False
        state.n_action = 0
    elif (phase == TREASURE_PHASE):
        state.need_treasure_phase = False
    elif (phase == BUY_PHASE):
        state.need_buy_phase = False
        state.n_buy = 0


def pay_cost(state, piece):
    state.n_buy = state.n_buy - 1
    state.n_coin = state.n_coin - piece.cost


def handle_acquisition(state, player, piece, destination):
    """
    Manage the secondary effects of transfering a Piece from the
    Supply to a Player's Hand: pay costs and track empty piles.
    """
    source = state.piles[piece.total_order_index]
    transfer_piece(piece, destination, source)
    if not(source):
        state.n_empty_piles = state.n_empty_piles + 1
        if piece.is_province:
            state.province_pile_empty = True


def gain(state, actor, piece, destination=None):
    destination = actor.DISCARD if (destination is None) else destination
    handle_acquisition(state, player=actor, piece=piece, destination=destination)


def buy_piece(state, actor, piece, destination=None):
    destination = actor.DISCARD if (destination is None) else destination
    pay_cost(state, piece)
    handle_acquisition(state, player=actor, piece=piece, destination=destination)


def play_piece(state, actor, piece, source=None, free=False):
    if not(free):
        state.n_action = state.n_action - 1
    source = actor.HAND if (source is None) else source
    transfer_piece(piece, actor.PLAY, source)
    resolve_effects(state, actor, piece)


def play_treasures(state, actor, pieces):
    for piece in pieces:
        transfer_piece(piece, actor.PLAY, actor.HAND)
        resolve_effects(state, actor, piece)
    end_phase(state, actor, TREASURE_PHASE)


def replenish_deck(actor):
    """
    When an Actor is meant to interact with more cards than are
    currently in their Deck, they must first shuffle their
    Discard pile (if any) and put it under the cards in their
    Deck before that interaction happens.
    """
    limbo = []
    while actor.DECK:
        transfer_top_piece(destination=limbo, source=actor.DECK)
    actor.DECK.extend(actor.DISCARD)
    actor.DISCARD.clear()
    shuffle(actor.DECK)
    while limbo:
        transfer_top_piece(destination=actor.DECK, source=limbo)


def validate_deck_size(actor, n):
    return (n <= len(actor.DECK))


def prepare_deck(actor, n):
    """
    Replenish Actor's Deck if necessary and if possible, prior to
    an interaction expecting at least n cards present.
    """
    if not(validate_deck_size(actor, n)):
        if actor.DISCARD:
            replenish_deck(actor)


def draw(state, actor, n):
    prepare_deck(actor, n)
    true_n = min(n, len(actor.DECK))
    for i in range(true_n):
        transfer_top_piece(destination=actor.HAND, source=actor.DECK)
    if true_n:
        actor.peek(actor.HAND[-true_n:])


def swap_top_cards_of_deck(state, actor, topcards):
    """
    Used by Sentry when there are at least two cards
    in actor's Deck, the top two cards have different
    piece types, and actor wants to swap the top card
    with the next card.
    """
    limbo = []
    transfer_top_piece(destination=limbo, source=actor.DECK)
    transfer_top_piece(destination=limbo, source=actor.DECK)
    actor.DECK.extend(limbo)


def trash(state, actor, piece, source=None):
    destination = state.TRASH
    source = actor.HAND if (source is None) else source
    transfer_piece(piece, destination, source)


def trash_pieces(state, actor, pieces, source=None):
    for piece in pieces:
        trash(state, actor, piece, source)


def reveal(state, actor, piece):
    pass


def reveal_pieces(state, actor, pieces):
    pass


def do_nothing(state, actor):
    """
    Represent choices which have no impact on the game, such as
    choosing not to trash any Treasures with Mine.
    """
    pass


def put(state, actor, piece, destination=None, source=None):
    destination = actor.HAND if (destination is None) else destination
    source = actor.DECK if (source is None) else source
    transfer_piece(piece, destination, source)


def set_aside(state, actor, piece, source):
    transfer_piece(piece, actor.ASIDE, source)


def topdeck(state, actor, piece, source):
    """ Place a card on top of actor's Deck. """
    transfer_piece(piece, destination=actor.DECK, source=source)


def discard(state, actor, piece, source=None):
    source = actor.HAND if (source is None) else source
    to_discard, to_keep = partition(lambda x: (x is piece), source)
    source.clear()
    source.extend(to_keep)
    actor.DISCARD.extend(to_discard)


def discard_pieces(state, actor, pieces, source=None):
    to_keep, to_discard = [], []
    source = actor.HAND if (source is None) else source
    for piece in source:
        if any((piece_to_discard is piece) for piece_to_discard in pieces):
            to_discard.append(piece)
        else:
            to_keep.append(piece)
    source.clear()
    source.extend(to_keep)
    actor.DISCARD.extend(to_discard)


def discard_aside(state, actor):
    source = actor.ASIDE
    actor.DISCARD.extend(source)
    source.clear()



######################
# Rules / Heuristics #
######################
def would_defeat(player, opponent):
    """
    NOTE
    Actor.__lt__ is defined according to Dominion's win condition.
    """
    return ((opponent < player) and not(player < opponent))


def would_defeat_or_tie(player, opponent):
    """ See also: will_beat and Actor.__lt__ """
    return not(player < opponent)


def modify_collection(player, acquisition):
    # Force the Player's collection property to reflect a
    # hypothetical post-Acquisition state without actually
    # altering any zones or piles.
    player.include.append(acquisition.gained_piece)
    if (hasattr(acquisition, 'lost_piece') and (acquisition.lost_piece is not None)):
        player.exclude.append(acquisition.lost_piece)


def revert_collection(player):
    # Ensure any temporary modifications to the Player's collection
    # are reverted.
    player.include.clear()
    player.exclude.clear()


def would_win_outright(player, acquisition):
    modify_collection(player, acquisition)
    result = all(would_defeat(player, opponent) for opponent in player.opponents)
    revert_collection(player)
    return result


def would_win(player, acquisition):
    """ Considers a tie for first place as a win. """
    modify_collection(player, acquisition)
    result = all(would_defeat_or_tie(player, opponent) for opponent in player.opponents)
    revert_collection(player)
    return result


def classify_consequences(consequences):
    """
    Strategy selection methods need to be able to differentiate
    between the types of Consequences a Decision generates.
    """
    return classify([PassOption, NullOption, Purchase, DependentAcquisition, Acquisition, Consequence],
                    consequences)


def extract_acquisitions(consequences):
    """
    Strategy selection methods need to be able to differentiate
    between the types of Consequences a Decision generates.
    """
    return classify([PassOption, NullOption, Acquisition, Consequence], consequences)


def classify_acts(acts):
    """
    Facilitate the naive look-ahead heuristic by partitioning the
    Acts a Strategy is choosing between into two groups:
    gainers:  contains an action which will potentially generate
              Consequences which are Acquisitions, and therefore,
              might be able to trigger the end of the game.
    neutrals: does not contain an action which will potentially
              generate Consequences which are Acquisitions, and
              therefore, will not be able to trigger the end of
              the game when played.
    """
    gainers, neutrals = partition(lambda act: (act.action.is_potential_terminator), acts)
    return gainers, neutrals


def evaluate_acquisitions(player, acquisitions):
    """
    Facilitate the naive look-ahead heuristic by partitioning the
    acquisitions a Strategy is choosing between into four groups:
    wins:     triggers the end of the game and leaves Strategy in a
              position where they are winning outright.
    ties:     triggers the end of the game and leaves Strategy in a
              position where they are tied for first.
    losses:   triggers the end of the game and leaves Strategy in a
              position where they are not in first place.
    neutrals: does not trigger the end of the game.
    """
    ends_game, neutrals = partition(player.state.acquisition_ends_game, acquisitions)
    wins, will_not_win_outright = partition(partial(would_win_outright, player), ends_game)
    ties, losses = partition(partial(would_win, player), will_not_win_outright)
    return wins, ties, losses, neutrals



class Effect:
    """
    Encode the intention to produce a pre-defined collection of
    changes to the game's state.
    """
    __slots__ = "function", "kwargs"
    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = dict(kwargs)

    def __call__(self, state):
        self.function(state, **self.kwargs)


class Initiate(Effect):
    """
    Distinguishes Effects which elicit behavior from one or more
    players during their resolution as opposed to producing a
    pre-defined collection of changes to the game's state.

    Example:
    When a Player resolves the effects of Council Room, simple effects
    cause them to draw 4 cards and add 1 buy. In contrast, the final
    effect causes each other player to draw a card. Similarly, at least
    one of an Attack's effects force the opponents of a player, in turn
    order, to perform a predefined procedure when resolved. Instances of
    Initiate, rather than of Effect, are used in such cases.

    Note: When decider is None, the default is for the actor to make
    the decision.
    """
    __slots__ = tuple()
    def __init__(self, actor, decision, decider=None):
        super().__init__(resolve,
                         actor=actor,
                         decision=decision,
                         decider=decider)


class Update(Effect):
    """
    Used by Processes to update their internal state and
    thereby track the impact of their main_effect over
    repeated iterations.
    """
    __slots__ = tuple()
    def __call__(self, state):
        self.function(**self.kwargs)


class Process:
    """
    A customizable looping construct used to encode the intention to
    change the state of the game a number of times that may be
    dynamically determined.

    Example:
    Draw until you have X cards in hand.

    Supports optional setup and teardown around the loop. Resolution
    of the setup Subprocess may alter the internal state of the Process
    in order to prevent the main Subprocess from running.

    Example:
    One application is implementing pairs of Effects where mandatory
    resolution of the second Effect is contingent on the outcome of
    resolving the first. Attacks are Processes with a setup Subprocess
    which will signal that a victim is immune if they reveal a Moat,
    preventing the main Subprocess encoding the attack itself from ever
    unfolding.
    """
    __slots__ = "actor", "main_effect", "setup_effect", "teardown_effect", "count", "max_count"
    def __init__(self, actor, main_effect, setup_effect=None, teardown_effect=None, max_count=0):
        self.actor = actor
        self.main_effect = main_effect
        self.setup_effect = setup_effect
        self.teardown_effect = teardown_effect
        self.count = 0
        self.max_count = max_count

    @property
    def count_antecedent(self):
        """
        Support antecedents which factor in the number of iterations
        performed relative to the maximum allowed.
        """
        return self.count < self.max_count

    @property
    def antecedent(self):
        raise NotImplementedError("Subclasses must define their own antecedent property.")

    def clear_internal_state(self):
        """
        Used by Processes which depend on internal state to compute
        the antecedent; by default, calibrates the main loop counter.
        """
        self.count = 0

    def increment_count(self):
        """ Hook for Subprocesses to directly control the number of iterations. """
        self.count = self.count + 1

    def setup(self, state):
        if (self.setup_effect is not None):
            self.setup_effect(state)

    def main(self, state):
        self.main_effect(state)

    def teardown(self, state):
        if (self.teardown_effect is not None):
            self.teardown_effect(state)
        self.clear_internal_state()

    def __call__(self, state):
        self.setup(state)
        while self.antecedent:
            self.main(state)
        self.teardown(state)



class Consequence:
    """ A collection of one or more related Effects or Processes. """
    __slots__ = "effects", "custom_message"
    def __init__(self, *effects, custom_message=None):
        self.effects = list(effects)
        self.custom_message = custom_message

    def add(self, effect):
        self.effects.append(effect)

    def __iter__(self):
        return iter(self.effects)


class PassOption(Consequence):
    """
    Inform a Strategy's choice of selection method by supporting
    classification of Consequences.
    """
    __slots__ = tuple()
    def __init__(self, actor, phase):
        super().__init__(Effect(end_phase, actor=actor, phase=phase))


class NullOption(Consequence):
    """
    Inform a Strategy's choice of selection method by supporting
    classification of Consequences.
    """
    __slots__ = tuple()
    def __init__(self, actor, custom_message=None):
        super().__init__(Effect(do_nothing, actor=actor), custom_message=custom_message)


class Act(Consequence):
    """
    Represents the possibility of playing an action Piece.
    The action attribute supports Strategy selection methods.
    """
    __slots__ = "action"
    def __init__(self, actor, action):
        super().__init__(Effect(play_piece, actor=actor, piece=action))
        self.action = action


class Acquisition(Consequence):
    """
    Represents the possibility of gaining a Piece from the Supply.
    May contain one or more additional Effects unrelated to gaining.
    The gained_piece attribute supports Strategy selection methods.
    """
    __slots__ = "gained_piece"
    def __init__(self, *effects):
        super().__init__(*effects)
        self.gained_piece = self.effects[0].kwargs['piece']


class Purchase(Acquisition):
    """
    Acquisition via buying.
    """
    __slots__ = tuple()
    def __init__(self, actor, piece):
        super().__init__(Effect(buy_piece, actor=actor, piece=piece))


class DependentAcquisition(Acquisition):
    """
    Represents the possibility of gaining a Piece from the Supply,
    contingent upon trashing one or more Pieces. May contain one
    or more additional Effects unrelated to the dependent gaining.
    The lost_piece attribute supports Strategy selection methods.
    """
    __slots__ = "lost_piece"
    def __init__(self, *effects):
        super().__init__(*effects)
        self.gained_piece = self.effects[1].kwargs['piece']
        self.lost_piece = self.effects[0].kwargs['piece']


class Decision:
    """
    A Decision encodes the logic which governs the generation of
    the distinct Consequences entailed by a specific Phase unfolding,
    by a specific Process unfolding, or by the resolution of an
    Effect.

    To play the game is to repeatedly engage in the following process
    until the game is terminated: [1] Determine the possible moves in
    a specific context; [2] Select one; and, [3] transform the state
    accordingly.

    Decisions are responsible for accomplishing [1]. Players are
    responsible for [2]. The engine handles [3].

    In a given context, the moves available to a player are derived
    from a prototypical set of routines defined and prescribed by the
    rules as a function of that context. To determine the full set of
    possible moves is to identify the functionally distinct ways of
    binding any free variables in the routines.

    Example:
    During the Action Phase the prototypical instruction set defines a
    routine of playing an Action card and a routine which passes the
    phase. There is no free variable to bind in the second routine. It
    isa sequence of operations on state variables which is fully
    defined a priori and independent of player input.

    In contrast, the free variable to bind in the routine of playing an
    Action card is the identity of a card satisfying the constraints.
    The full set of possibilities is the union of O and A, where:
    O is the singleton set containing the routine for passing the Action
    Phase; and, A is a set where each element is the routine of playing
    an Action Card with its free variable bound to a functionally
    distinct card satisfying the constraints (i.e., two different types
    of Action cards in hand).
    """
    def __init__(self):
        pass

    def generate_choices(self, state, actor):
        return []


class Subprocess(Decision):
    """
    A Subprocess is a Decision used by a Process which optionally
    adds Updates to each of the Consequences it generates. When a
    Consequence is selected and enacted, the Update is enacted as
    part of the default routine of enacting all of the Consequence's
    Effects. Enacting an Update causes it to alter the internal state
    of the Process controlling the Subprocess which added it to the
    chosen Consequence. In this way, feedback about the number of times
    certain types of Consequences have been selected is provided to the
    master Process, allowing dynamic control over the number of times it
    forces actors to choose from amongst the Consequences generated by
    its Subprocesses. To support this method of communication, a
    reference to the Process which instantiated the Subprocess is
    required.
    """
    __slots__ = "process"
    def __init__(self, process):
        self.process = process


class ImmunityProcedure(Subprocess):
    """ Reveal Moat to be unaffected by an Attack. """
    __slots__ = "success_message", "failure_message"
    def __init__(self, process):
        super().__init__(process=process)
        self.success_message = [
            "become immune to the attack by revealing Moat.",
            "becomes immune to the attack by revealing Moat.",
        ]
        self.failure_message = [
            "fail to defend yourself.",
            "fails to defend themselves.",
        ]

    def generate_choices(self, state, actor):
        signal = Update(self.process.mark_immunity)
        moat = get_pieces(actor.HAND,
                          unique=True,
                          predicate=lambda x: repr(x) == "Moat")
        if moat:
            return [Consequence(Effect(reveal, actor=actor, piece=moat[0]),
                                signal,
                                custom_message=self.success_message)]
        return [NullOption(actor, custom_message=self.failure_message)]


class Attack(Process):
    """ Processes which check for victim immunity. """
    __slots__ = "immunity"
    def __init__(self, actor, main_effect, teardown_effect=None, max_count=1):
        super().__init__(actor=actor,
                         main_effect=main_effect,
                         setup_effect=Initiate(actor, ImmunityProcedure(self)),
                         teardown_effect=teardown_effect,
                         max_count=max_count)
        self.immunity = False

    def mark_immunity(self):
        self.immunity = True

    def clear_internal_state(self):
        super().clear_internal_state()
        self.immunity = False

    @property
    def antecedent(self):
        """ Subclasses check a conjunction of this antecedent with their own. """
        return not(self.immunity)


class OneshotAttack(Attack):
    """
    Processes which check for victim immunity and only force victims
    to perform a procedure once.
    """
    __slots__ = tuple()
    @property
    def antecedent(self):
        """ Not immune and yet to have performed the victim procedure. """
        return super().antecedent and super().count_antecedent



class ActionChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [PassOption(actor, ACTION_PHASE)]
        if state.n_action:
            actions = get_pieces(actor.HAND,
                                 unique=True,
                                 predicate=lambda x: x.is_action)
            for action in actions:
                choices.append(Act(actor, action))
        return choices


class TreasureChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [PassOption(actor, TREASURE_PHASE)]
        treasures = get_pieces(actor.HAND,
                               unique=False,
                               predicate=lambda x: x.is_treasure)
        if treasures:
            choices.append(Consequence(Effect(play_treasures, actor=actor, pieces=treasures)))
        return choices


class BuyChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [PassOption(actor, BUY_PHASE)]
        if state.n_buy:
            buyables = state.filter_supply(lambda x: x.cost <= state.n_coin)
            for buyable in buyables:
                choices.append(Purchase(actor, buyable))
        return choices


class MineChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [NullOption(actor)]
        trashables = get_pieces(actor.HAND,
                                unique=True,
                                predicate=lambda x: x.is_treasure)
        for trashable in trashables:
            max_cost = trashable.cost + 3
            gainables = state.filter_supply(lambda x: x.is_treasure and (x.cost <= max_cost))
            if gainables:
                for gainable in gainables:
                    choices.append(DependentAcquisition(Effect(trash,
                                                               actor=actor,
                                                               piece=trashable),
                                                        Effect(gain,
                                                               actor=actor,
                                                               piece=gainable,
                                                               destination=actor.HAND)))
            else:
                choices.append(Consequence(Effect(trash, actor=actor, piece=trashable)))
        return choices


class CellarChoices(Decision):
    __slots__ = tuple()
    def generate_heuristic_choices(self, state, actor):
        pieces_to_discard = get_pieces(actor.HAND, unique=False, predicate=lambda x: x.is_victory)
        if pieces_to_discard:
            return [Consequence(Effect(discard_pieces, actor=actor, pieces=pieces_to_discard),
                                Effect(draw, actor=actor, n=len(pieces_to_discard)))]
        return []

    def generate_human_choices(self, state, actor):
        choices = []
        discardables = actor.HAND
        if discardables:
            combos = get_piece_combinations(discardables, kmin=1)
            for combo in combos:
                # QoL: List pieces alphabetically within combinations.
                combo = sorted(combo, key=lambda p: repr(p))
                choices.append(Consequence(Effect(discard_pieces,
                                                  actor=actor,
                                                  pieces=combo),
                                           Effect(draw,
                                                  actor=actor,
                                                  n=len(combo))))
        # QoL: List choices in increasing order of # of pieces to discard;
        #      within the choices sharing # of pieces to discard, list
        #      alphabetically by first piece.
        choices.sort(key=lambda c: (c.effects[-1].kwargs['n'],
                                    repr(c.effects[0].kwargs['pieces'][0])))
        return choices

    def generate_choices(self, state, actor):
        choices = [NullOption(actor)]
        # Case: Full range of discard choices.
        if repr(actor) == 'You':
            choices.extend(self.generate_human_choices(state, actor))
        # Case: Strategy Heuristic: Do nothing or discard all Victory cards.
        else:
            choices.extend(self.generate_heuristic_choices(state, actor))
        return choices


class RemodelChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = []
        trashables = get_pieces(actor.HAND, unique=True)
        if trashables:
            for trashable in trashables:
                gainables = state.filter_supply(lambda x: x.cost <= (trashable.cost + 2))
                if gainables:
                    for gainable in gainables:
                        choices.append(DependentAcquisition(Effect(trash, actor=actor, piece=trashable),
                                                            Effect(gain, actor=actor, piece=gainable)))
                else:
                    choices.append(Consequence(Effect(trash, actor=actor, piece=trashable)))
        else:
            choices.append(NullOption(actor))
        return choices


class MilitiaVictimProcedure(Subprocess):
    """ Discard a card in your Hand. """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = []
        discardables = get_pieces(actor.HAND, unique=True)
        for discardable in discardables:
            choices.append(Consequence(Effect(discard,
                                              actor=actor,
                                              piece=discardable)))
        return choices


class MilitiaAttack(Attack):
    """ Discard until there are 3 cards in your Hand. """
    __slots__ = tuple()
    def __init__(self, actor):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, MilitiaVictimProcedure(self)))

    @property
    def antecedent(self):
        """ Not immune and more than 3 cards Hand. """
        return super().antecedent and (len(self.actor.HAND) > 3)


class MilitiaChoices(Decision):
    def generate_choices(self, state, actor):
        consequence = Consequence()
        for victim in actor.opponents:
            consequence.add(MilitiaAttack(victim))
        consequence.custom_message = ["attack with Militia!", "attacks with Militia!"]
        return [consequence]


class WorkshopChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        gainables = state.filter_supply(lambda x: x.cost <= 4)
        if gainables:
            return [Acquisition(Effect(gain, actor=actor, piece=gainable)) for gainable in gainables]
        return [NullOption(actor)]



class ChapelChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [NullOption(actor)]
        trashables = actor.HAND
        if trashables:
            trash_choices = []
            combos = get_piece_combinations(trashables, kmin=1, kmax=4)
            for combo in combos:
                # QoL: Sort the pieces within alphabetically.
                combo = sorted(combo, key=lambda p: repr(p))
                trash_choices.append(Consequence(Effect(trash_pieces, actor=actor, pieces=combo)))
            # QoL: Sort the choices descending by size and within each
            #      size, alphabetically.
            trash_choices.sort(key=lambda c: (len(c.effects[0].kwargs['pieces']),
                                              repr(c.effects[0].kwargs['pieces'][0])))
            choices.extend(trash_choices)
        return choices


class HarbingerChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [NullOption(actor)]
        targets = get_pieces(actor.DISCARD, unique=True)
        for target in targets:
            choices.append(Consequence(Effect(topdeck, actor=actor, piece=target, source=actor.DISCARD)))
        return choices


class VassalChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = []
        prepare_deck(actor=actor, n=1)
        if not(actor.DECK):
            choices.append(NullOption(actor))
        else:
            topcard = actor.DECK[-1]
            # The option of discarding the top card is always available.
            choices.append(Consequence(Effect(discard,
                                              actor=actor,
                                              piece=topcard,
                                              source=actor.DECK)))

            # An action adds the option of playing the top card from discard.
            # To simplify, don't actually move it into Discard, just play it
            # from the top of the deck.
            if topcard.is_action:
                choices.append(Consequence(Effect(play_piece,
                                                  actor=actor,
                                                  piece=topcard,
                                                  source=actor.DECK,
                                                  free=True)))
        return choices


class BureaucratProcedure(Subprocess):
    """
    Reveal a Victory card from hand and topdeck it.
    If you can't, reveal hand.
    """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        signal = Update(self.process.increment_count)
        victory_cards = get_pieces(actor.HAND,
                                   unique=True,
                                   predicate=lambda x: x.is_victory)
        if not(victory_cards):
            if not(actor.HAND):
                choice = NullOption(actor)
                choice.add(signal)
                return [choice]
            else:
                return [Consequence(Effect(reveal_pieces, actor=actor, pieces=actor.HAND),
                                    signal)]
        else:
            choices = []
            for victory_card in victory_cards:
                choices.append(Consequence(Effect(reveal, actor=actor, piece=victory_card),
                                           Effect(topdeck, actor=actor, piece=victory_card, source=actor.HAND),
                                           signal))
        return choices


class BureaucratAttack(OneshotAttack):
    """
    Each other player reveals a Victory card from their hand
    and puts it onto their deck (or reveals a hand with no
    Victory cards).
    """
    __slots__ = tuple()
    def __init__(self, actor):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, BureaucratProcedure(self)))


class BureaucratChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        silver = state.filter_supply(lambda x: repr(x) == "Silver")
        if silver:
            outcome = Acquisition(Effect(gain,
                                         actor=actor,
                                         piece=silver[-1],
                                         destination=actor.DECK))
        else:
            outcome = Consequence()
        for victim in actor.opponents:
            outcome.add(BureaucratAttack(victim))
            outcome.custom_message = ["attack with Bureaucrat!", "attacks with Bureaucrat!"]
        return [outcome]


class MoneylenderChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [NullOption(actor)]
        coppers = get_pieces(actor.HAND,
                             unique=True,
                             predicate=lambda x: repr(x) == "Copper")
        if coppers:
            choices.append(Consequence(Effect(trash, actor=actor, piece=coppers[0]),
                                       Effect(add_coin, n=3)))
        return choices



class PoacherProcedure(Subprocess):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        signal = Update(self.process.increment_count)
        choices = []
        discardables = get_pieces(actor.HAND, unique=True)
        for discardable in discardables:
            discard_effect = Effect(discard, actor=actor, piece=discardable)
            choices.append(Consequence(discard_effect, signal))
        return choices


class PoacherProcess(Process):
    __slots__ = tuple()
    def __init__(self, actor, max_count):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, PoacherProcedure(self)),
                         setup_effect=None,
                         teardown_effect=None,
                         max_count=max_count)

    @property
    def antecedent(self):
        """
        At least one card left to discard and at least one card available.
        """
        assert super().count_antecedent and len(self.actor.HAND)


class PoacherChoices(Decision):
    """
    Discard a card per empty Supply pile.
    """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        n_to_discard = state.n_empty_piles
        n_to_discard = min(len(actor.HAND), n_to_discard)
        if not(n_to_discard):
            return [NullOption(actor)]
        return [Consequence(PoacherProcess(actor=actor, max_count=n_to_discard))]


class ThroneRoomChoices(Decision):
    """
    You may play an Action card from your hand twice.
    """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        choices = [NullOption(actor)]
        actions = get_pieces(actor.HAND,
                             unique=True,
                             predicate=lambda x: x.is_action)
        for action in actions:
            choices.append(Consequence(Effect(play_piece, actor=actor, piece=action, source=actor.HAND, free=True),
                                       Effect(resolve_effects, actor=actor, piece=action)))
        return choices


class BanditProcedure(Subprocess):
    """
    Each other player reveals the top 2 cards of their deck,
    trashes a revealed Treasure other than Copper, and discards
    the rest.
    """
    __slots__ = tuple()
    def could_trash(self, piece):
        """ Custom predicate for filtering affected Pieces. """
        return piece.is_treasure and (repr(piece) != "Copper")

    def generate_choices(self, state, actor):
        signal = Update(self.process.increment_count)
        prepare_deck(actor, n=2)
        if not(actor.DECK):
            option = NullOption(actor)
            option.add(signal)
            return [option]

        decksize = len(actor.DECK)
        if (decksize == 1):
            topcard = actor.DECK[-1]
            effect_function = trash if self.could_trash(topcard) else discard
            return [Consequence(Effect(reveal, actor=actor, piece=topcard),
                                Effect(effect_function, actor=actor, piece=topcard, source=actor.DECK),
                                signal)]
        else:
            choices = []
            topcards = [actor.DECK[-1], actor.DECK[-2]]
            #reveal_effects = [Effect(reveal, actor=actor, piece=card) for card in topcards]
            reveal_effect = Effect(reveal_pieces, actor=actor, pieces=topcards)
            trashables, discardables = partition(self.could_trash, topcards)
            unique_trashables = get_pieces(trashables, unique=True)
            n_unique_trashables = len(unique_trashables)
            # Case: Covers both cards being trashable with distinct identities.
            if (n_unique_trashables == 2):
                index_permutations = [(0, 1), (1, 0)]
                for index_permutation in index_permutations:
                    trash_index, discard_index = index_permutation
                    trashable = unique_trashables[trash_index]
                    discardable = unique_trashables[discard_index]
                    choices.append(Consequence(reveal_effect,
                                               Effect(trash,
                                                      actor=actor,
                                                      piece=trashable,
                                                      source=actor.DECK),
                                               Effect(discard,
                                                      actor=actor,
                                                      piece=discardable,
                                                      source=actor.DECK),
                                                signal))
            # Case: Covers one card being trashable and one being discardable;
            #       or, both cards being trashable while sharing a piece type.
            elif (n_unique_trashables == 1):
                trashable = unique_trashables[0]
                consequence = Consequence(reveal_effect,
                                          Effect(trash,
                                                 actor=actor,
                                                 piece=trashable,
                                                 source=actor.DECK))
                if discardables:
                    discardable = discardables[0]
                    consequence.add(Effect(discard,
                                           actor=actor,
                                           piece=discardable,
                                           source=actor.DECK))
                consequence.add(signal)
                choices.append(consequence)
            # Case: Nothing to trash, discard both pieces.
            else:
                choices.append(Consequence(reveal_effect,
                                           Effect(discard_pieces,
                                                  actor=actor,
                                                  pieces=discardables,
                                                  source=actor.DECK),
                                           signal))
        return choices


class BanditAttack(OneshotAttack):
    __slots__ = tuple()
    def __init__(self, actor):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, BanditProcedure(self)))


class BanditChoices(Decision):
    """
    Gain a Gold. Each other player reveals the top 2 cards
    of their deck, trashes a revealed Treasure other than
    Copper, and discards the rest.
    """
    def generate_choices(self, state, actor):
        gold = state.filter_supply(lambda x: repr(x) == "Gold")
        if gold:
            outcome = Acquisition(Effect(gain, actor=actor, piece=gold[-1]))
            custom_message = ["gain a Gold then attack with Bandit!",
                              "gains a Gold then attacks with Bandit!"]
        else:
            outcome = Consequence(Effect(do_nothing, actor=actor))
            custom_message = ["attack with Bandit!", "attacks with Bandit!"]
        for victim in actor.opponents:
            outcome.add(BanditAttack(victim))
        outcome.custom_message = custom_message
        return [outcome]


class CouncilRoomProcedure(Subprocess):
    """ Draw a card. """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        return [Consequence(Effect(draw, actor=actor, n=1),
                            Update(self.process.increment_count))]

class CouncilRoomProcess(Process):
    """ Each other player draws a card. """
    __slots__ = tuple()
    def __init__(self, actor):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, CouncilRoomProcedure(self)))

    @property
    def antecedent(self):
        return self.count < self.max_count


class CouncilRoomChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        interaction = Consequence()
        for victim in actor.opponents:
            interaction.add(CouncilRoomProcess(victim))
        interaction.custom_message = ["prompt each other player to draw a card with Council Room.",
                                      "prompts each other player to draw a card with Council Room."]
        return [interaction]


class LibraryProcedure(Subprocess):
    """
    Put the top card of your deck into your hand; however, if
    it is an action, you can set it aside instead.
    """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        prepare_deck(actor=actor, n=1)
        if not(actor.DECK):
            return [NullOption(actor)]
        else:
            topcard = actor.DECK[-1]
            choices = [Consequence(Effect(put, actor=actor, piece=topcard))]
            if topcard.is_action:
                choices.append(Consequence(Effect(set_aside,
                                                  actor=actor,
                                                  piece=topcard,
                                                  source=actor.DECK)))
            return choices


class LibraryTeardown(Subprocess):
    """ Discard any actions set aside during the Library Process. """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        print("DEBUG | len(actor.HAND): {}".format(len(actor.HAND)))
        if not(actor.ASIDE):
            return [NullOption(actor)]
        return [Consequence(Effect(discard_aside, actor=actor))]


class LibraryProcess(Process):
    """
    Repeatedly require an actor to make a choice generated by
    LibraryProcedure so long as the preconditions hold.
    Afterwards, discard skipped cards using LibraryTeardown.
    """
    __slots__ = tuple()
    def __init__(self, actor):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, LibraryProcedure(self)),
                         setup_effect=None,
                         teardown_effect=Initiate(actor, LibraryTeardown(self)))

    @property
    def antecedent(self):
        """
        Hand size < 7 and at least one card about which to decide.
        """
        handsize = len(self.actor.HAND)
        n_targets = len(self.actor.DECK) + len(self.actor.DISCARD)
        return ((handsize < 7) and n_targets)


class LibraryChoices(Decision):
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        process = LibraryProcess(actor)
        if process.antecedent:
            consequence = Consequence(process)
            consequence.custom_message = [
                "begin the process described by Library.",
                "begins the process described by Library.",
             ]
            return [consequence]
        return [NullOption(actor)]


class SentryChoices(Decision):
    """
    Look at the top 2 cards of your deck.
    Trash and/or discard any number of them.
    Put the rest back on top in any order.
    """
    def generate_choices(self, state, actor):
        prepare_deck(actor, n=2)
        if not(actor.DECK):
            return [NullOption(actor)]

        # Represents looking at whatever is there and doing nothing else.
        # The other options will show human players what is on top so no
        # need for a "look at the top card(s)" Effect.
        choices = [NullOption(actor)]
        decksize = len(actor.DECK)
        topcards = [actor.DECK[-1]] if (decksize == 1) else [actor.DECK[-1], actor.DECK[-2]]

        non_null_choices = []
        # Do the same thing (trash or discard) to the topcard(s).
        functions = [trash, discard]
        for function in functions:
            consequence = Consequence()
            for topcard in topcards:
                consequence.add(Effect(function, actor=actor, piece=topcard, source=actor.DECK))
            non_null_choices.append(consequence)
        non_null_choices.sort(key=lambda c: (repr(c.effects[0].function.__name__)))
        choices.extend(non_null_choices)

        unique_topcards = get_pieces(topcards, unique=True)
        n_unique_topcards = len(unique_topcards)

        # Trashing one then discarding the other; or,
        # Trashing one and leaving the other on top; or,
        # Discarding one and leaving the other on top; or,
        # Leaving both on top but swapping their positions.
        if (n_unique_topcards == 2):
            non_null_choices = []
            choices.append(Consequence(Effect(swap_top_cards_of_deck,
                                              actor=actor,
                                              topcards=unique_topcards)))
            index_permutations = [(0, 1), (1, 0)]
            for index_permutation in index_permutations:
                trash_index, discard_index = index_permutation
                non_null_choices.append(Consequence(Effect(trash, actor=actor, piece=topcards[trash_index], source=actor.DECK),
                                           Effect(discard, actor=actor, piece=topcards[discard_index], source=actor.DECK)))
                non_null_choices.append(Consequence(Effect(trash, actor=actor, piece=topcards[trash_index], source=actor.DECK)))
                non_null_choices.append(Consequence(Effect(discard, actor=actor, piece=topcards[discard_index], source=actor.DECK)))
            non_null_choices.sort(key=lambda c: (repr(c.effects[0].function.__name__)))
            choices.extend(non_null_choices)

        # QoL: Sort the order choices are displayed by number of cards involved.
        choices.sort(key=lambda c: len(c.effects))
        return choices


class WitchProcedure(Subprocess):
    """ Gain a Curse if able. """
    __slots__ = tuple()
    def generate_choices(self, state, actor):
        signal = Update(self.process.increment_count)
        gainables = state.filter_supply(lambda x: repr(x) == "Curse")
        if gainables:
            option = Acquisition(Effect(gain, actor=actor, piece=gainables[0]),
                                 signal)
        else:
            option = NullOption(actor)
        option.add(signal)
        return [option]


class WitchAttack(OneshotAttack):
    """ Each other player gains a Curse. """
    __slots__ = tuple()
    def __init__(self, actor):
        super().__init__(actor=actor,
                         main_effect=Initiate(actor, WitchProcedure(self)))



class WitchChoices(Decision):
    def generate_choices(self, state, actor):
        outcome = Consequence()
        for victim in actor.opponents:
            outcome.add(WitchAttack(victim))
        outcome.custom_message = ["attack with Witch!", "attacks with Witch!"]
        return [outcome]


class ArtisanChoices(Decision):
    def generate_choices(self, state, actor):
        choices = []
        gainables = state.filter_supply(lambda x: x.cost <= 5)
        placeables = get_pieces(actor.HAND, unique=True)
        if not(gainables):
            if not(placeables):
                choices.append(NullOption(actor))
            else:
                for placeable in placeables:
                    choices.append(Consequence(Effect(topdeck, actor=actor, piece=placeable, source=actor.HAND)))
        elif not(placeables):
            for gainable in gainables:
                choices.append(Acquisition(Effect(gain, actor=actor, piece=gainable, destination=actor.HAND),
                                           Effect(topdeck, actor=actor, piece=gainable, source=actor.HAND)))
        else:
            placeable_tois = set([placeable.total_order_index for placeable in placeables])
            for gainable in gainables:
                gainable_toi = gainable.total_order_index
                for placeable in placeables:
                    choices.append(Acquisition(Effect(gain, actor=actor, piece=gainable, destination=actor.HAND),
                                               Effect(topdeck, actor=actor, piece=placeable, source=actor.HAND)))
                # Case: Gaining a Piece with a novel type relative to Pieces in Hand
                #       implies the option to place a Piece of that type has not yet
                #       been generated (otherwise it will have been, so avoid duplication).
                if (gainable_toi not in placeable_tois):
                    choices.append(Acquisition(Effect(gain, actor=actor, piece=gainable, destination=actor.HAND),
                                               Effect(topdeck, actor=actor, piece=gainable, source=actor.HAND)))

        return choices
