import inspect
import numpy as np

from evolvedominion.params import (
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
    N_PROCESSES,
)
from evolvedominion.engine.pieces import (
    Piece,
    Curse,
    Estate,
    Duchy,
    Province,
    Copper,
    Silver,
    Gold,
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
from evolvedominion.engine.engine import (
    enact,
    resolve,
    resolve_effects,
    expand_choices,
    transfer_piece,
    transfer_top_piece,
    NullOption,
    PassOption,
    Act,
    Acquisition,
    Purchase,
    DependentAcquisition,
    MilitiaAttack,
    would_defeat,
    would_defeat_or_tie,
    classify_consequences,
    would_win,
    would_win_outright,
)
from evolvedominion.algorithm.evolve import (
    Genome,
    Phenotype,
    Simulation,
    random_initial_sigma,
    recombine_sigma,
)
from evolvedominion.agents.player import Player, EchoPlayer
from evolvedominion.agents.strategy import RandomStrategy, EchoRandomStrategy, Strategy, EchoStrategy
from evolvedominion.display.echo import EchoSession
from evolvedominion.engine.session import (
    _DEFAULT_KINGDOM,
    _ANOTHER_KINGDOM,
    Session,
)


_VICTORY_AND_TREASURE = [
    Curse,
    Estate,
    Duchy,
    Province,
    Copper,
    Silver,
    Gold,
]

ZONE_NAMES = [
    "HAND",
    "DECK",
    "DISCARD",
    "ASIDE",
    "PLAY",
]
N_ZONES = len(ZONE_NAMES)

N_PILES = 17
N_COPPER = 60
N_SILVER = 40
N_GOLD = 30
N_KINGDOM_CARD = 10
N_CURSE = 30
N_VICTORY_CARD = 12
N_ESTATE = 24
INITIAL_HAND_SIZE = 5
N_STARTING_COPPER = 7
N_STARTING_ESTATE = 3

ZONE_ATTRIBUTES = {"piles", "TRASH", "supply"}


class CustomError(Exception):
    def __init__(self, **kwargs):
        super().__init__(self.get_message(**kwargs))

    def get_message(self, **kwargs):
        return "TEST [{}]\n"


class AttributeDidChangeError(CustomError):
    """ An attribute changed which wasn't predicted to change. """
    def get_message(self, **kwargs):
        caller = kwargs['caller']
        attribute = kwargs['attribute']
        old_value = kwargs['old']
        new_value = kwargs['new']
        start = super().get_message().format(caller)
        end = "Did not predict attribute [{}] to change from [{}] to [{}].".format(attribute, old_value, new_value)
        return "".join([start, end])


class AttributeDidNotChangeError(CustomError):
    """ An attribute predicted to change did not. """
    def get_message(self, **kwargs):
        caller = kwargs['caller']
        attribute = kwargs['attribute']
        start = super().get_message().format(caller)
        end = "Attribute [{}] was predicted to change, but did not".format(attribute)
        return "".join([start, end])


class ValuePredictionError(CustomError):
    """ Incorrect prediction for the magnitude of a value change. """
    def get_message(self, **kwargs):
        caller = kwargs['caller']
        attribute = kwargs['attribute']
        value = kwargs['value']
        actual = kwargs['actual']
        start = super().get_message().format(caller)
        end = "Attribute [{}] has value: [{}] instead of [{}]".format(attribute, value, actual)
        return "".join([start, end])


def extract_state_data(state):
    """
    Store the relevant attribute values of a State instance
    to facilitate testing the impact of Effects.
    Note: Trash and other zone-related changes are handled
          elsewhere.
    """
    data = {}
    for slot in state.__slots__:
        if (slot not in ZONE_ATTRIBUTES):
            value = getattr(state, slot)
            if isinstance(value, list):
                value = list(value)
            data[slot] = value
    return data


def extract_player_zone_data(player):
    """
    Store copies of a player's zones to support testing effects
    which change the location of Pieces.
    """
    player_zone_data = {}
    for i in range(N_ZONES):
        ZONE_NAME = ZONE_NAMES[i]
        player_zone_data[ZONE_NAME] = list(getattr(player, ZONE_NAME))
    return player_zone_data


def extract_supply(state):
    copied_supply = []
    for pile in state.piles:
        copied_supply.append(list(pile))
    return copied_supply


def extract_zone_data(state):
    """
    Store copies of the zones of each player to support testing
    effects which change the location of Pieces.
    """
    zone_data = {}
    zone_data['TRASH'] = list(state.TRASH)
    zone_data['supply'] = extract_supply(state)
    for player in state.players:
        zone_data[player.pid] = extract_player_zone_data(player)
    return zone_data


def compare_player_zone_data(player_zone_data_t1, player_zone_data_t0):
    """ Return a list of Zones which don't match. """
    changed_zone_names = []
    for key in player_zone_data_t1:
        if not(zones_are_equal(player_zone_data_t1, player_zone_data_t0)):
            changed_zone_names.append(key)
    return changed_zone_names


def compare_zone_data(zone_data_t1, zone_data_t0):
    """ Check for matching zone data across all players. """
    results = {}
    for key in zone_data_t1:
        if (key not in ZONE_ATTRIBUTES):
            player_zone_data_t1 = zone_data_t1[key]
            player_zone_data_t0 = zone_data_t0[key]
            results[key] = compare_player_zone_data(player_zone_data_t1,
                                                    player_zone_data_t0)
    return results


def validate_supply(supply_t1, supply_t0):
    """ Ensure the supply is the same at t1 and t0. """
    invalid_ids = []
    for i, pile in enumerate(supply_t1):
        if not(zones_are_equal(supply_t1[i], supply_t0[i])):
            invalid_ids.append(i)
    if invalid_ids:
        raise ValueError("Supply pile[{}] differed at t1 relative to t0.".format(i))
    return True


def validate_trash(trash_t1, trash_t0):
    """ Ensure the trash is the same at t1 and t0. """
    if not(zones_are_equal(trash_t1, trash_t0)):
        raise ValueError("Trash contains different pieces at t1 compared to t0.")
    return True


def validate_zone_data(zone_data_t1, zone_data_t0):
    """
    Ensure which Pieces exist and where they are is the same
    at t1 as t0.
    """
    if validate_trash(zone_data_t1['TRASH'], zone_data_t0['TRASH']):
        if validate_supply(zone_data_t1['supply'], zone_data_t0['supply']):
            comparison = compare_zone_data(zone_data_t1, zone_data_t0)
            offenders = []
            for player_pid in comparison:
                player_result = comparison[player_pid]
                if player_result:
                    offenders.append(player_pid)
            try:
                assert not(offenders)
            except AssertionError:
                print("Players with the following PIDs had mismatched zones: {}.".format(offenders))


def attributes_which_differ(data_t1, data_t0):
    """
    data_t1 and data_t0 are dictionaries with keys
    the attributes of a State instance and values the
    corresponding attribute value at times t1 and t0
    respectively. Returns the keys with different
    values, in order to isolate the impact of the
    Effect which transformed the State at t0 into
    the State at t1.
    """
    changed_attributes = []
    for key in data_t1:
        if (data_t1[key] != data_t0[key]):
            changed_attributes.append(key)
    return changed_attributes


def _difference_bool(value_t1, value_t0):
    """
    Return 1 if value started False and became True
    Return 0 if value stayed the same
    Return -1 if value started True and became False
    """
    if (value_t1 == value_t0):
        return 0
    elif (value_t1 and not(value_t0)):
        return 1
    return -1


def _difference_int(value_t1, value_t0):
    return value_t1 - value_t0


def difference(value_t1, value_t0, type=int):
    """
    Compute the magnitude of a change to an integer or boolean
    state attribute value.
    """
    if (type is int):
        return _difference_int(value_t1, value_t0)
    elif (type is bool):
        return _difference_bool(value_t1, value_t0)
    else:
        raise TypeError("{} can't be compared with difference.".format(type))


def zones_are_equal(zone0, zone1):
    """ Comparisons only care about identity of elements. """
    assert len(zone0) == len(zone1)
    zone0_ids = sorted([id(piece) for piece in zone0])
    zone1_ids = sorted([id(piece) for piece in zone1])
    return zone0_ids == zone1_ids


def quantify_state_changes(data_t1, data_t0):
    """
    Compute the magnitude of changes to integer and boolean
    state attribute values.
    """
    changes = {}
    changed_attributes = attributes_which_differ(data_t1, data_t0)
    for attribute in changed_attributes:
        value_t1, value_t0 = data_t1[attribute], data_t0[attribute]
        type_t1, type_t0 = type(value_t1), type(value_t0)
        assert type_t1 is type_t0
        changes[attribute] = difference(value_t1, value_t0, type_t1)
    return changes


def validate_state_changes(data_t1, data_t0, **changes):
    """ Ensure the only differences in the State are those predicted. """
    caller = inspect.stack()[1].function
    state_changes = quantify_state_changes(data_t1, data_t0)
    for attribute, value in changes.items():
        # Attribute predicted to change did so.
        try:
            assert attribute in state_changes
        except AssertionError:
            raise AttributeDidNotChangeError(caller=caller, attribute=attribute)
        # Attribute predicted to change did so in the predicted manner.
        try:
            actual = state_changes[attribute]
            assert actual == value
        except AssertionError:
            raise ValuePredictionError(caller=caller, attribute=attribute, value=value, actual=actual)
    # Only predicted changes occurred.
    try:
        for changed_attribute in state_changes:
            assert changed_attribute in changes
    except AssertionError:
        old_value = data_t0[changed_attribute]
        new_value = data_t1[changed_attribute]
        raise AttributeDidChangeError(caller=caller, attribute=changed_attribute, old=old_value, new=new_value)


def get_test_session(echo=False, kingdom=_DEFAULT_KINGDOM):
    if echo:
        player_type = Player
        strategy_type = EchoRandomStrategy
        session_type = EchoSession
    else:
        player_type = RandomStrategy
        strategy_type = RandomStrategy
        session_type = Session

    players = [player_type(0)] + [strategy_type(i) for i in range(1, 4)]
    session = session_type(kingdom)
    session.accept_players(players)
    session.initialize_state()
    return session


def get_strategy_session(kingdom=_DEFAULT_KINGDOM):
    """
    Use Simulation to spawn Strategy instances with
    randomly initialized Phenotypes / Genomes to
    facilitate testing selection methods.
    """
    simulation = Simulation(simname="test", N=8)
    simulation.random_spawn()
    simulation.update_players()
    players = simulation.players[:4]
    session = Session()
    session.accept_players(players)
    session.initialize_state()
    return session


def get_supply_data(kingdom):
    """
    Determine the specification of the Supply given a list
    of Kingdom cards to use.
    """
    supply_data = {}
    supply_data[Curse] = N_CURSE
    supply_data[Estate] = N_ESTATE - (N_STARTING_ESTATE * 4)
    supply_data[Duchy] = N_VICTORY_CARD
    supply_data[Province] = N_VICTORY_CARD
    supply_data[Copper] = N_COPPER - (N_STARTING_COPPER * 4)
    supply_data[Silver] = N_SILVER
    supply_data[Gold] = N_GOLD
    for piece in kingdom:
        supply_data[piece] = N_KINGDOM_CARD
    return supply_data


def _validate_initial_supply(state, supply_data):
    """
    Ensure the Supply is initialized correctly given
    a specification.
    """
    # Number of piles is correct.
    assert len(state.piles) == N_PILES

    for pile_index, (piece, pile_size) in enumerate(supply_data.items()):
        pile = state.piles[pile_index]
        # Every piece pile contains is the correct type.
        assert all(isinstance(element, piece) for element in pile)
        # Contains the correct number of Pieces.
        assert len(pile) == pile_size


def _validate_initial_player_pieces(state):
    """
    Ensure the distribution of Pieces across Players is correct.
    """

    for player in state.players:
        # Correct number of cards in Hand
        assert len(player.HAND) == INITIAL_HAND_SIZE
        collection = player.collection
        n_copper, n_estate = 0, 0
        for card in collection:
            if isinstance(card, Copper):
                n_copper += 1
            elif isinstance(card, Estate):
                n_estate += 1

        # Correct number of Coppers and Estates provided
        assert n_copper == N_STARTING_COPPER
        assert n_estate == N_STARTING_ESTATE
        assert len(collection) == (N_STARTING_COPPER + N_STARTING_ESTATE)
        other_zones = player.zones[2:]
        other_zones.append(player.include)
        other_zones.append(player.exclude)

        # No cards in any zone that isn't Deck or Hand, nor in the
        # temporary zones used by modify_collection.
        assert all((len(zone) == 0) for zone in other_zones)


def _validate_initial_player_win_condition_attributes(state):
    """
    Ensure the attributes used to determine the ranking of
    Players are correct in the initial state.
    """
    INITIAL_VICTORY_POINTS = 3
    INITIAL_N_TURNS_PLAYED = 0
    for player in state.players:
        # victory_points property returns the correct value
        assert player.victory_points == INITIAL_VICTORY_POINTS
        # n turns played calibrated
        assert player.n_turns_played == INITIAL_N_TURNS_PLAYED


def _validate_initial_player_state(session):
    """
    Ensure each player in a session has a reference to
    the correct state.
    """
    for player in session.players:
        assert (player.state is session.state)


def _validate_player_opponents(player, players):
    """
    Ensure that the opponents of a player have the correct
    identities and that they occur in turn-order, which
    is essential for Attacks and other interactive Effects.
    """
    opponents = player.opponents
    player_id = id(player)
    player_position = players.index(player)
    other_players = list(filter(lambda x: (x is not player), players))
    other_player_ids = set([id(other_player) for other_player in other_players])

    # Correct number of unique opponents.
    assert len(other_player_ids) == (len(players) - 1)

    # Correct opponent identities.
    assert not(player_id in other_player_ids)

    first = players[0]
    second = players[1]
    third = players[2]
    fourth = players[3]

    if (player_position == 0):
        turn_ordered_opponents = [second, third, fourth]
    elif (player_position == 1):
        turn_ordered_opponents = [third, fourth, first]
    elif (player_position == 2):
        turn_ordered_opponents = [fourth, first, second]
    elif (player_position == 3):
        turn_ordered_opponents = [first, second, third]

    # Correctly ordered opponents.
    assert all((opponents[i] is turn_ordered_opponents[i]) for i in range(3))


def _validate_initial_player_opponents(state):
    """
    Ensure that each player's opponents are correct and
    occur in turn order starting with the player who comes
    immediately after them in turn order.
    """
    players = state.players
    for player in players:
        _validate_player_opponents(player, players)



def test_initial_conditions(echo=False, kingdom=_DEFAULT_KINGDOM):
    """ Ensure that the initial state is correct. """
    INITIAL_HAND_SIZE = 5
    N_STARTING_COPPER = 7
    N_STARTING_ESTATE = 3
    ESTATE_PILE_INDEX = 1
    PROVINCE_PILE_INDEX = 3
    COPPER_PILE_INDEX = 4

    S = get_test_session(echo=echo, kingdom=kingdom)
    state = S.state
    supply_data = get_supply_data(kingdom=kingdom)

    _validate_initial_supply(state=state, supply_data=supply_data)
    _validate_initial_player_pieces(state=state)
    _validate_initial_player_win_condition_attributes(state=state)
    _validate_initial_player_state(session=S)
    _validate_initial_player_opponents(state=state)


def test_termination_properties(echo=False, kingdom=_DEFAULT_KINGDOM):
    """
    According to the rules of Dominion, State.game_over is true when:
        >= 3 piles in the Supply are empty; OR,
        the Province pile in the Supply is empty.

    To facilitate the evolution of strategies, a cap on the maximum
    number of turns played is also factored in.

    State.almost_over is true when:
        2 piles in the Supply are empty; OR,
        the Province pile has a single piece in it.
    """
    PROVINCE_PILE_INDEX = 3
    MAX_N_TURNS = 180
    S = get_test_session(echo=echo, kingdom=kingdom)
    state = S.state

    # Neither should be true in the initial state.
    assert not(state.game_over)
    assert not(state.almost_over)

    state.n_empty_piles = 2
    assert not(state.game_over)
    assert state.almost_over

    state.province_pile_empty = True
    assert state.game_over

    state.n_empty_piles = 4
    assert state.game_over

    state.province_pile_empty = False
    assert state.game_over

    state.n_empty_piles = 0
    state.piles[PROVINCE_PILE_INDEX] = [Province()]
    assert not(state.game_over)
    assert state.almost_over

    state.n_total_turns_played = 181
    assert state.game_over


def validate_effects(state, effects, **changes):
    """
    Take a snapshot of the state before and after applying the
    effects. Determine whether the changes are precisely those
    described by **changes.
    """
    data_t0 = extract_state_data(state)
    enact(state, effects)
    data_t1 = extract_state_data(state)
    validate_state_changes(data_t1, data_t0, **changes)


def validate_null_option(choices):
    """
    Common case where the only Consequence generated
    by a Decision is the option to do nothing.
    """
    assert len(choices) == 1
    assert isinstance(choices[0], NullOption)


def valid_draw(pid, n, zd_t0):
    t0_hand = zd_t0[pid]['HAND']
    t0_deck = zd_t0[pid]['DECK']
    for i in range(n):
        transfer_top_piece(destination=t0_hand, source=t0_deck)





def test_cellar_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    cellar = Cellar()
    predicted_changes = {
        'n_action':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, cellar.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_moat_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    current_player = state.current_player
    current_player_pid = current_player.pid
    moat = Moat()

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, moat.simple_effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[current_player_pid]['HAND']
    t0_deck = zone_data_t0[current_player_pid]['DECK']

    transfer_top_piece(destination=t0_hand, source=t0_deck)
    transfer_top_piece(destination=t0_hand, source=t0_deck)

    # Test the impact of Moat's simple effects on the zones of
    # every player.
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_merchant_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    current_player = state.current_player
    current_player_pid = current_player.pid
    merchant = Merchant()
    predicted_changes = {
        'n_action':1,
        'merchant_silver_bonus':1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, merchant.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[current_player_pid]['HAND']
    t0_deck = zone_data_t0[current_player_pid]['DECK']

    transfer_top_piece(destination=t0_hand, source=t0_deck)

    # Test the impact of Merchant's simple effects on the zones
    # of every player.
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_village_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    current_player = state.current_player
    current_player_pid = current_player.pid
    village = Village()
    predicted_changes = {
        'n_action':2,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, village.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[current_player_pid]['HAND']
    t0_deck = zone_data_t0[current_player_pid]['DECK']

    transfer_top_piece(destination=t0_hand, source=t0_deck)

    # Test the impact of Village's simple effects on the zones
    # of every player.
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_smithy_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    current_player = state.current_player
    current_player_pid = current_player.pid
    smithy = Smithy()

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, smithy.simple_effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[current_player_pid]['HAND']
    t0_deck = zone_data_t0[current_player_pid]['DECK']

    transfer_top_piece(destination=t0_hand, source=t0_deck)
    transfer_top_piece(destination=t0_hand, source=t0_deck)
    transfer_top_piece(destination=t0_hand, source=t0_deck)

    # Test the impact of Smithy's simple effects on the zones
    # of every player.
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_militia_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    militia = Militia()
    predicted_changes = {
        'n_coin':2,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, militia.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_market_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    current_player = state.current_player
    current_player_pid = current_player.pid
    market = Market()
    predicted_changes = {
        'n_action':1,
        'n_buy':1,
        'n_coin':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, market.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[current_player_pid]['HAND']
    t0_deck = zone_data_t0[current_player_pid]['DECK']

    transfer_top_piece(destination=t0_hand, source=t0_deck)

    # Test the impact of Market's simple effects on the zones of
    # every player.
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_copper_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    copper = Copper()
    predicted_changes = {
        'n_coin':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, copper.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_gold_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    gold = Gold()
    predicted_changes = {
        'n_coin':3,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, gold.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_silver_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    silver = Silver()
    predicted_changes = {
        'n_coin':2,
        'n_silver_played_this_turn':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, silver.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_silver_and_merchant_interactions():
    S = get_test_session()
    S.start_turn()
    state = S.state
    silver = Silver()

    # Turn on the merchant silver bonus.
    # The first time a Silver is played this turn it should add
    # 3 coin instead of 2 coin.
    state.merchant_silver_bonus = 1
    predicted_changes = {
        'n_coin':3,
        'n_silver_played_this_turn':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, silver.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Keeping the merchant silver bonus on, play another one.
    # This time it should add 2 instead of 3, like usual,
    # since it isn't the first Silver played this turn.
    predicted_changes = {
        'n_coin':2,
        'n_silver_played_this_turn':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, silver.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_remodel_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    remodel = Remodel()

    # Case: Nothing in hand to Trash. Only choice generated
    # is doing nothing, which has no impact on the State.
    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    assert not(actor.HAND)
    choices = expand_choices(state, actor, remodel.decision)
    assert len(choices) == 1
    only_choice = choices[0]
    assert isinstance(only_choice, NullOption)
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, only_choice.effects)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Case: Something in hand to Trash, but nothing available
    # to gain given its cost and the state of the Supply.
    dummy_piece = Piece(cost=-5)
    actor.HAND.append(dummy_piece)
    choices = expand_choices(state, actor, remodel.decision)
    assert len(choices) == 1
    only_choice = choices[0]
    effects = only_choice.effects
    assert len(effects) == 1
    assert not(isinstance(effects[0], DependentAcquisition))
    assert effects[0].function.__name__ == "trash"
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_trash = zone_data_t0['TRASH']
    transfer_top_piece(destination=t0_trash, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Clear the trash.
    state.TRASH.clear()

    # Case: Something in hand to Trash, multiple choices for
    # what to gain given its cost and the state of the Supply.
    dummy_province = Province()
    actor.HAND.append(dummy_province)
    choices = expand_choices(state, actor, remodel.decision)
    # Since a Province has the highest cost of any Piece in the
    # Base Set, and none of the Supply's piles are empty, the
    # number of choices should be equal to the number of piles.
    assert len(choices) == len(state.piles)

    # Each one should be a DependentAcquisition
    assert all(isinstance(choice, DependentAcquisition) for choice in choices)

    # Choose one.
    choice_index = 8
    choice = choices[choice_index]
    effects = choice.effects

    assert len(effects) == 2

    # Verify gained_piece attribute works.
    gained_piece = choice.gained_piece
    pile_index = gained_piece.total_order_index
    assert choice.gained_piece is state.piles[pile_index][-1]

    # Verify lost_piece attribute works.
    assert choice.lost_piece is dummy_province

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_trash = zone_data_t0['TRASH']
    t0_pile = zone_data_t0['supply'][pile_index]

    transfer_top_piece(destination=t0_trash, source=t0_hand)
    transfer_top_piece(destination=t0_discard, source=t0_pile)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_cellar_choices():
    """ Testing the heuristic cellar choices. """
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    cellar = Cellar()

    # Case: No victory cards in hand; only choice is
    # to do nothing.
    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    assert not(actor.HAND)

    choices = expand_choices(state, actor, cellar.decision)
    assert len(choices) == 1
    only_choice = choices[0]
    assert isinstance(only_choice, NullOption)
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, only_choice.effects)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Case: Victory cards in hand; two choices: do nothing,
    # or discard all of the victory cards and draw that many.
    dummy_estate = Estate()
    dummy_duchy = Duchy()
    dummy_province = Province()
    actor.HAND.extend([dummy_estate, dummy_duchy, dummy_province])
    n_to_discard = n_to_draw = 3

    choices = expand_choices(state, actor, cellar.decision)
    assert len(choices) == 2
    discard_nothing_choice = choices[0]
    assert isinstance(only_choice, NullOption)
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, only_choice.effects)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)

    discard_three_choice = choices[1]
    effects = discard_three_choice.effects
    assert len(effects) == 2
    discard_effect = effects[0]
    draw_effect = effects[1]

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    for i in range(n_to_discard):
        transfer_top_piece(destination=t0_discard, source=t0_hand)
    for i in range(n_to_draw):
        transfer_top_piece(destination=t0_hand, source=t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_workshop_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    workshop = Workshop()

    # Case: Something to gain.
    choices = expand_choices(state, actor, decision=workshop.decision)
    workshop_max_cost = 4
    matching_piles = [pile for pile in state.piles if (pile and (pile[-1].cost <= workshop_max_cost))]
    assert len(choices) == len(matching_piles)

    # Choose one.
    choice_index = 10
    choice = choices[choice_index]
    assert isinstance(choice, Acquisition)

    effects = choice.effects
    assert len(effects) == 1

    gained_piece = choice.gained_piece
    pile_index = gained_piece.total_order_index

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_pile = zone_data_t0['supply'][pile_index]

    transfer_top_piece(destination=t0_discard, source=t0_pile)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Case: Nothing to gain.
    state._initialize_empty_piles()
    choices = expand_choices(state, actor, decision=workshop.decision)
    only_choice = choices[0]
    assert isinstance(only_choice, NullOption)
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, only_choice.effects)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_mine_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    mine = Mine()

    # Case: Nothing in hand to Trash. Only choice generated
    # is doing nothing, which has no impact on the State.
    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    assert not(actor.HAND)
    choices = expand_choices(state, actor, mine.decision)
    assert len(choices) == 1
    only_choice = choices[0]
    assert isinstance(only_choice, NullOption)
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, only_choice.effects)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Case: Something in hand to Trash, but nothing available
    # to gain given its cost and the state of the Supply;
    # since Mine's effect is optional, the option to do nothing
    # should be offered along with the option to trash the piece
    # for nothing in return.
    dummy_piece = Piece(cost=-5)
    dummy_piece.is_treasure = True
    actor.HAND.append(dummy_piece)
    choices = expand_choices(state, actor, mine.decision)
    assert len(choices) == 2
    pass_choice = choices[0]
    assert isinstance(pass_choice, NullOption)

    trash_choice = choices[1]
    effects = trash_choice.effects
    assert len(effects) == 1
    assert not(isinstance(effects[0], DependentAcquisition))
    assert effects[0].function.__name__ == "trash"
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_trash = zone_data_t0['TRASH']
    transfer_top_piece(destination=t0_trash, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Clear the trash.
    state.TRASH.clear()

    # Case: Something in hand to Trash, multiple choices for
    # what to gain given its cost and the state of the Supply.
    gold = Gold()
    actor.HAND.append(gold)
    choices = expand_choices(state, actor, mine.decision)
    # Since Gold is the most expensive Treasure and there's nothing
    # missing from the Supply, the number of choices should be equal
    # to the number of Treasure types, 3, plus 1, since Mine's effect
    # is always optional.
    assert len(choices) == 4

    # Each one after the first should be a DependentAcquisition.
    assert isinstance(choices[0], NullOption)
    assert all(isinstance(choice, DependentAcquisition) for choice in choices[1:])

    # Choose one.
    choice_index = 3
    choice = choices[choice_index]
    effects = choice.effects
    assert len(effects) == 2
    gained_piece = choice.gained_piece
    pile_index = gained_piece.total_order_index
    # Verify gained_piece attribute works.
    assert gained_piece is state.piles[pile_index][-1]

    # Verify lost_piece attribute works.
    assert choice.lost_piece is gold

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_trash = zone_data_t0['TRASH']
    t0_pile = zone_data_t0['supply'][pile_index]

    transfer_top_piece(destination=t0_trash, source=t0_hand)
    transfer_top_piece(destination=t0_hand, source=t0_pile)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_action_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    decision = S.decisions[ACTION_PHASE]

    # Case: State.n_action < 1, so no options.
    state.n_action = 0
    choices = expand_choices(state, actor, decision)
    assert len(choices) == 1
    only_choice = choices[0]
    assert isinstance(only_choice, PassOption)

    # Case: State.n_action > 0, but no Actions in hand.
    state.n_action = 1
    choices = expand_choices(state, actor, decision)
    assert len(choices) == 1
    only_choice = choices[0]
    assert isinstance(only_choice, PassOption)

    # Case: State.n_action > 0 and an Action in hand.
    state.n_action = 1
    village = Village()
    actor.HAND.append(village)
    choices = expand_choices(state, actor, decision)
    # The option to pass the action phase should be given
    # along with the option to play Village.
    assert len(choices) == 2
    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)
    act_choice = choices[1]
    assert isinstance(act_choice, Act)
    # Verify action attribute works
    assert act_choice.action is village

    predicted_changes = {
        'n_action':1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, act_choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_play = zone_data_t0[actor_pid]['PLAY']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    t0_play.append(t0_hand.pop())
    transfer_top_piece(destination=t0_hand, source=t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Should still be the action phase afterward.
    assert state.need_action_phase

    predicted_changes = {
        'n_action':-2,
        'need_action_phase':-1,
    }

    # Apply the PassOption to test ending the action phase.
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, pass_choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_treasure_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    decision = S.decisions[TREASURE_PHASE]

    state.need_action_phase = False
    state.need_treasure_phase = True

    # Case: No treasures in hand, only option
    # is to pass the Treasure Phase.
    actor.HAND.clear()
    choices = expand_choices(state, actor, decision)
    assert len(choices) == 1
    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)

    # Case: Treasures in hand, options are:
    # pass the Treasure Phase, or play all of them.
    dummy_treasures = [Copper(), Silver(), Gold()]
    n_coin_to_gain = 1 + 2 + 3
    actor.HAND.extend(dummy_treasures)
    choices = expand_choices(state, actor, decision)
    assert len(choices) == 2
    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)
    play_choice = choices[1]
    assert play_choice.effects[0].function.__name__ == "play_treasures"

    # Play the treasures.
    predicted_changes = {
        'n_coin':n_coin_to_gain,
        'n_silver_played_this_turn':1,
        'need_treasure_phase':-1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, play_choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_play = zone_data_t0[actor_pid]['PLAY']
    transfer_top_piece(destination=t0_play, source=t0_hand)
    transfer_top_piece(destination=t0_play, source=t0_hand)
    transfer_top_piece(destination=t0_play, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Should not be the treasure phase afterward; revert that.
    assert not(state.need_treasure_phase)
    state.need_treasure_phase = True

    predicted_changes = {
        'need_treasure_phase':-1,
    }

    # Apply the PassOption to test ending the treasure phase.
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, pass_choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_buy_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    decision = S.decisions[BUY_PHASE]

    state.need_action_phase = False
    state.need_treasure_phase = False
    state.need_buy_phase = True

    # Case: Can't afford anything in the supply;
    # only option is to pass the buy phase.
    state.n_coin = -5
    state.n_buy = 1
    choices = expand_choices(state, actor, decision)
    assert len(choices) == 1
    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)

    # Case: Can afford something in the supply, but
    # no buys left for the turn.
    state.n_coin = 0
    state.n_buy = 0
    choices = expand_choices(state, actor, decision)
    assert len(choices) == 1
    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)

    # Case: Can afford more than one thing in the supply,
    # and have at least 1 buy left for the turn.
    state.n_coin = 4
    state.n_buy = 1
    choices = expand_choices(state, actor, decision)

    # Since passing the phase is always available, the first
    # choice should still be a PassOption.
    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)

    # The rest should be Purchases.
    assert all(isinstance(choice, Purchase) for choice in choices[1:])

    # Choose one.
    choice_index = 7
    purchase = choices[choice_index]

    # Verify gained_piece attribute.
    gained_piece = purchase.gained_piece
    pile_index = gained_piece.total_order_index
    assert gained_piece is state.piles[pile_index][-1]

    # Buy it.
    cost = gained_piece.cost
    predicted_changes = {
        'n_coin':-cost,
        'n_buy':-1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, purchase.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_pile = zone_data_t0['supply'][pile_index]
    transfer_top_piece(destination=t0_discard, source=t0_pile)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Test passing the buy phase.
    predicted_changes = {
        'need_buy_phase':-1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, pass_choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_militia_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents
    opponent_pids = [opponent.pid for opponent in opponents]

    militia = Militia()
    moat = Moat()

    # Make room for Moat in opponents[0]'s hand.
    transfer_top_piece(destination=opponents[0].DISCARD, source=opponents[0].HAND)
    opponents[0].HAND.append(moat)

    # Make opponents[1] already only have 3 cards in hand.
    transfer_top_piece(destination=opponents[1].DISCARD, source=opponents[1].HAND)
    transfer_top_piece(destination=opponents[1].DISCARD, source=opponents[1].HAND)

    # Arrange opponents[2] to have a hand full of Villages.
    opponents[2].DISCARD.extend(opponents[2].HAND)
    opponents[2].HAND.clear()
    opponents[2].HAND.extend([Village(), Village(), Village(), Village(), Village()])

    choices = expand_choices(state, actor, decision=militia.decision)

    # There should only be one choice: the one which unfolds the process of attacking
    # each player.
    assert len(choices) == 1
    choice = choices[0]

    # There should be one instance of MilitiaAttack for each victim.
    effects = choice.effects
    assert len(effects) == 3
    assert all(isinstance(effect, MilitiaAttack) for effect in effects)

    # Allow the attack to unfold.
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    # Nothing should happen to opponents[0] because they will
    # reveal Moat to make themselves immune.
    # Nothing should happen to opponents[1] because they are
    # already at 3 cards in hand so can't discard further.
    # Opponents[2] will randomly discard 2 cards, which must be Villages.
    t0_opp2_hand = zone_data_t0[opponent_pids[2]]['HAND']
    t0_opp2_discard = zone_data_t0[opponent_pids[2]]['DISCARD']
    transfer_piece(piece=zone_data_t1[opponent_pids[2]]['DISCARD'][-1],
                   destination=t0_opp2_discard, source=t0_opp2_hand)
    transfer_piece(piece=zone_data_t1[opponent_pids[2]]['DISCARD'][-2],
                   destination=t0_opp2_discard, source=t0_opp2_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_actor_sorting():
    """
    Test Actor.__lt__ for sorting players based
    on the Dominion win conditions, and the auxillary
    methods which rely on it for direct comparisons.
    """
    p0 = RandomStrategy(0)
    p1 = RandomStrategy(1)

    assert p0.victory_points == p1.victory_points
    assert p0.n_turns_played == p1.n_turns_played
    assert not(p0 < p1) and not(p1 < p0)
    assert not(would_defeat(p0, p1))
    assert not(would_defeat(p1, p0))
    assert would_defeat_or_tie(p0, p1)
    assert would_defeat_or_tie(p1, p0)

    province = Province()
    p0.HAND.append(province)

    assert p0.victory_points > p1.victory_points
    assert p0.n_turns_played == p1.n_turns_played
    assert not(p0 < p1) and (p1 < p0)
    assert would_defeat(p0, p1)
    assert not(would_defeat(p1, p0))
    assert would_defeat_or_tie(p0, p1)
    assert not(would_defeat_or_tie(p1, p0))

    p0.n_turns_played = p0.n_turns_played + 1
    assert p0.victory_points > p1.victory_points
    assert p0.n_turns_played > p1.n_turns_played
    assert not(p0 < p1) and (p1 < p0)
    assert would_defeat(p0, p1)
    assert not(would_defeat(p1, p0))
    assert would_defeat_or_tie(p0, p1)
    assert not(would_defeat_or_tie(p1, p0))

    province2 = Province()
    p1.HAND.append(province2)
    assert p0.victory_points == p1.victory_points
    assert p0.n_turns_played > p1.n_turns_played
    assert (p0 < p1) and not(p1 < p0)
    assert not(would_defeat(p0, p1))
    assert would_defeat(p1, p0)
    assert not(would_defeat_or_tie(p0, p1))
    assert would_defeat_or_tie(p1, p0)


def test_session_award_players():
    """
    Ensure players are awarded in relation to how well
    they placed in a game of Dominion
    """
    S = get_test_session()
    while not(S.state.game_over):
        S.start_turn()
        S.action_phase()
        S.treasure_phase()
        S.buy_phase()
        S.end_turn()
    S.award_players()
    for i in range(3):
        j = i + 1
        assert would_defeat_or_tie(S.players[i], S.players[j])

    target_points = [3, 2, 1, 0]
    for i in range(4):
        assert (S.players[i].score == target_points[i])


def test_meta_select_pass_instead_of_forced_loss():
    """
    Ensure the naive look-ahead heuristic elects to
    do nothing when all non-nullary choices entail
    triggering the end of the game this turn while
    the Strategy isn't outright winning nor tied for
    first place.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # Manipulate the supply such that the only available
    # options are forced losses, except for the option
    # to pass.

    # In this contrived scenario, all of the Treasure piles
    # in the Supply have 1 card left, 2 other piles are empty,
    # and the current player has played more turns than everyone
    # else, although they share the same victory points.
    # The two cards in the player's hand are Mine and Gold.
    # The only safe play will be to use Mine's option to do nothing.

    copper_pile_index = 4
    silver_pile_index = 5
    gold_pile_index = 6

    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    mine = Mine()
    actor.HAND.append(mine)
    actor.HAND.append(Gold())
    # Empty the Curse pile.
    state.piles[0].clear()
    # Empty the Estate pile.
    state.piles[1].clear()
    state.n_empty_piles = 2

    assert state.almost_over

    state.piles[copper_pile_index] = [Copper()]
    state.piles[silver_pile_index] = [Silver()]
    state.piles[gold_pile_index] = [Gold()]
    choices = expand_choices(state, actor, mine.decision)
    # There should be four choices. 1 to pass, 1 to trash Gold and
    # gain the remaining treasure in each of the 3 treasure piles.
    assert len(choices) == 4
    assert isinstance(choices[0], NullOption)
    assert all(isinstance(choice, DependentAcquisition) for choice in choices[1:])

    passes, nulls, purchases, dep_acqs, acqs, other = classify_consequences(choices)
    pass_choice = nulls[0]
    # The default selection method in these cases is
    # actor.select_dependent_acquisition.
    selection_method = actor.select_dependent_acquisition
    result = actor.meta_select(acquisitions=dep_acqs,
                               pass_choice=pass_choice,
                               selection_method=selection_method)
    assert result is pass_choice


def test_meta_meta_select_pass_instead_of_forced_loss():
    """
    Ensure the naive look-ahead heuristic will pass
    the Action Phase instead of playing available
    Actions which exclusively generate consequences which
    will trigger the end of the game this turn while the
    Strategy isn't winning outright nor tied for first.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # In this contrived scenario, the current player is
    # losing, has 1 action left, and a Workshop in hand.
    # The supply is manipulated so that every choice of
    # acquisition via the Workshop will lead to a loss.
    # meta_meta_select should pass the action phase instead
    # of choosing to play Workshop.

    state.piles[0].clear()
    state.piles[-1].clear()
    state.n_empty_piles = 2

    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            new_pile = [topcard]
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    for opponent in opponents:
        opponent.HAND.append(Province())

    workshop = Workshop()
    actor.HAND.append(workshop)

    state.need_action_phase = True
    state.n_action = 1


    decision = S.decisions[ACTION_PHASE]
    choices = expand_choices(state, actor, decision)

    assert len(choices) == 2

    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)
    acts = choices[1:]
    assert isinstance(acts[0], Act)

    result = actor.meta_meta_select(acts=acts, pass_choice=pass_choice)
    assert result is pass_choice



def test_meta_select_lose_in_arbitrary_fashion():
    """
    Ensure the naive look-ahead heuristic chooses
    the first available option which loses when
    it has no other options; extensions to this
    behavior would improve it so that it selects
    the losing option which gets it closest to
    first place as possible.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # Manipulate the supply such that the only available
    # options are forced losses.

    # In this contrived scenario, all of the piles with cards which
    # are gainable via Workshop have a single card left in them,
    # 2 other piles are empty, and the current player has played more
    # turns than everyone else, although they share the same victory
    # points.

    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    workshop = Workshop()
    # Empty the final two piles.
    state.piles[-1].clear()
    state.piles[-2].clear()
    state.n_empty_piles = 2

    # Leave a single card in every pile with a piece gainable by
    # Workshop.
    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            if (topcard.cost <= 4):
                new_pile = [topcard]
            else:
                new_pile = list(pile)
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    choices = expand_choices(state, actor, workshop.decision)

    # Every choice should be an Acquisition.
    passes, nulls, purchases, dep_acqs, acqs, other = classify_consequences(choices)
    assert not(passes) and not(nulls) and not(purchases) and not(dep_acqs) and not(other)

    pass_choice = None

    # The default selection method in these cases is
    # actor.select_independent_acquisition.
    selection_method = actor.select_independent_acquisition
    result = actor.meta_select(acquisitions=acqs,
                               pass_choice=pass_choice,
                               selection_method=selection_method)

    # Making this Acquisition will trigger the end of the game this turn.
    assert state.acquisition_ends_game(result)

    # current_player will not be tied for first or outright winning after
    # making this Acquisition.
    assert not(would_win_outright(actor, result) and not(would_win(actor, result)))


def test_meta_select_find_tie():
    """
    Ensure that the naive look-ahead heuristic will
    over-ride evolve preference selection when it
    detects there is a way to trigger the end of the
    game this turn while it is tied for first place.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # In this contrived scenario, all of the players
    # have played the same number of turns. Two of the
    # current player's opponents have the same number of
    # victory points as the current player. One opponent
    # has a lead equivalent to a Province. Many piles have
    # only 1 piece left, including the Province pile.
    # The current player is in their Buy Phase, has 1 buy
    # left, and has enough coin to buy anything in the Supply.

    for player in state.players:
        player.n_turns_played = 5

    opponents[0].HAND.append(Province())

    state.need_action_phase = False
    state.need_treasure_phase = False
    state.need_buy_phase = True
    state.n_buy = 1
    state.n_coin = 10

    # Empty the final two piles.
    state.piles[-1].clear()
    state.piles[-2].clear()
    state.n_empty_piles = 2

    # Leave 1 piece in every non-empty pile the current player
    # can afford to buy from.
    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            if (topcard.cost <= state.n_coin):
                new_pile = [topcard]
            else:
                new_pile = list(pile)
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    decision = S.decisions[BUY_PHASE]
    choices = expand_choices(state, actor, decision)

    # The first choice should be to pass the buy phase.
    # The rest should be Purchase instances.
    passes, nulls, purchases, dep_acqs, acqs, other = classify_consequences(choices)
    pass_choice = passes[0]
    assert not(nulls) and not(dep_acqs) and not(acqs) and not(other)
    assert all(isinstance(choice, Purchase) for choice in choices[1:])

    # The default selection method in these cases is select_purchase_acquisition.
    selection_method = actor.select_purchase_acquisition

    result = actor.meta_select(acquisitions=purchases,
                               pass_choice=pass_choice,
                               selection_method=selection_method)

    assert state.acquisition_ends_game(result)
    assert not(would_win_outright(actor, result))
    # Note: would_win considers a tie for first as a 'win'.
    assert would_win(actor, result)

    # Demonstrate that this was the only option which accomplished
    # achieving a tie for first.
    for purchase in purchases:
        if (purchase is not result):
            if state.acquisition_ends_game(purchase):
                assert not(would_win_outright(actor, purchase))
                assert not(would_win(actor, purchase))



def test_meta_meta_select_find_tie():
    """
    Ensure that the naive look-ahead heuristic will
    override evolved preferences in order to select an
    Action which will generate a consequence which will
    trigger the end of the game this turn while the
    Strategy is tied for first.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # In this contrived scenario, the current player is
    # tied for second with 2 other opponents. Each player
    # has played the same number of turns, but the final
    # opponent has a lead equivalent to a Duchy.
    # The current player is in their Action Phase with 1
    # action to play, and has a Workshop and Remodel in
    # hand. The supply is manipulated so that every choice
    # of gaining from the supply will end the game but only
    # Remodeling the Workshop into a Duchy will tie the game.

    for player in state.players:
        player.n_turns_played = 5

    state.piles[0].clear()
    state.piles[-1].clear()
    state.n_empty_piles = 2

    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            new_pile = [topcard]
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    opponents[0].HAND.append(Duchy())

    workshop = Workshop()
    actor.HAND.append(workshop)
    remodel = Remodel()
    actor.HAND.append(remodel)

    state.need_action_phase = True
    state.n_action = 1

    decision = S.decisions[ACTION_PHASE]
    choices = expand_choices(state, actor, decision)

    # Pass the Action Phase, play Workshop, or play Remodel
    assert len(choices) == 3

    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)
    acts = choices[1:]
    assert all(isinstance(act, Act) for act in acts)

    result = actor.meta_meta_select(acts=acts, pass_choice=pass_choice)
    assert result.action == remodel


def test_meta_select_find_outright_win():
    """
    Ensure the naive look-ahead heuristic will override
    evolved preferences to select choices which will
    trigger the end of the game this turn while it is
    outright winning.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # In this contrived scenario, all of the players
    # have played the same number of turns. Two of the
    # current player's opponents have the same number of
    # victory points as the current player. One opponent
    # has a lead equivalent to an Estate. Many piles have
    # only 1 piece left, excluding the Province pile.
    # The current player is in their Buy Phase, has 1 buy
    # left, and has enough coin to buy anything in the Supply.
    # Purchasing Duchy is the only choice that wins outright.

    for player in state.players:
        player.n_turns_played = 5

    opponents[0].HAND.append(Estate())

    state.need_action_phase = False
    state.need_treasure_phase = False
    state.need_buy_phase = True
    state.n_buy = 1
    state.n_coin = 6

    # Empty the final two piles.
    state.piles[-1].clear()
    state.piles[-2].clear()
    state.n_empty_piles = 2

    # Leave 1 piece in every non-empty pile the current player
    # can afford to buy from.
    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            if (topcard.cost <= state.n_coin):
                new_pile = [topcard]
            else:
                new_pile = list(pile)
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    decision = S.decisions[BUY_PHASE]
    choices = expand_choices(state, actor, decision)

    # The first choice should be to pass the buy phase.
    # The rest should be Purchase instances.
    passes, nulls, purchases, dep_acqs, acqs, other = classify_consequences(choices)
    pass_choice = passes[0]
    assert not(nulls) and not(dep_acqs) and not(acqs) and not(other)
    assert all(isinstance(choice, Purchase) for choice in choices[1:])

    # The default selection method in these cases is select_purchase_acquisition.
    selection_method = actor.select_purchase_acquisition

    result = actor.meta_select(acquisitions=purchases,
                               pass_choice=pass_choice,
                               selection_method=selection_method)

    assert state.acquisition_ends_game(result)
    assert would_win_outright(actor, result)
    assert isinstance(result.gained_piece, Duchy)

    # Demonstrate that this was the only option which accomplished
    # an outright win.
    for purchase in purchases:
        if (purchase is not result):
            if state.acquisition_ends_game(purchase):
                assert not(would_win_outright(actor, purchase))



def test_meta_meta_select_find_outright_win_despite_tie():
    """
    Ensure the naive look-ahead heuristic will override
    evolved preferences to select any Action it finds can
    generate at least one consequence which will trigger the
    end of the game this turn while it is outright winning,
    even if it sees an opportunity to tie.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # In this contrived scenario, the current player is
    # tied for second with 2 other opponents. Each player
    # has played the same number of turns, but the final
    # opponent has a lead equivalent to a Duchy.
    # The current player is in their Action Phase with 1
    # action to play, and has a Mine, Gold, and Remodel in
    # hand. The supply is manipulated so that every choice
    # of gaining from the supply will end the game but only
    # Remodeling the Gold into a Province will win outright.
    # Remodeling the Gold or Mine into a Duchy will only tie.
    # Other choices lose.

    for player in state.players:
        player.n_turns_played = 5

    state.piles[0].clear()
    state.piles[-1].clear()
    state.n_empty_piles = 2

    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            new_pile = [topcard]
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    opponents[0].HAND.append(Duchy())

    gold = Gold()
    actor.HAND.append(gold)
    mine = Mine()
    actor.HAND.append(mine)
    remodel = Remodel()
    actor.HAND.append(remodel)

    state.need_action_phase = True
    state.n_action = 1

    decision = S.decisions[ACTION_PHASE]
    choices = expand_choices(state, actor, decision)

    # Pass the Action Phase, play Mine, or play Remodel.
    assert len(choices) == 3

    pass_choice = choices[0]
    assert isinstance(pass_choice, PassOption)
    acts = choices[1:]
    assert all(isinstance(act, Act) for act in acts)

    result = actor.meta_meta_select(acts=acts, pass_choice=pass_choice)
    assert result.action == remodel



def test_meta_select_find_outright_win_despite_tie():
    """
    Ensure the naive look-ahead heuristic will override
    evolved preferences to select choices which will
    trigger the end of the game this turn while it is
    outright winning, even if it sees the chance to
    tie first.
    """
    S = get_strategy_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents

    # In this contrived scenario, all of the players
    # have played the same number of turns. Two of the
    # current player's opponents have the same number of
    # victory points as the current player. One opponent
    # has a lead equivalent to a Duchy. All but two empty
    # piles have 1 piece left, including the Province pile.
    # The current player is in their Action Phase, and has
    # a Remodel, a Copper, and a Gold in their hand.
    # Remodeling the Copper loses. Remodeling the Gold into
    # a Duchy would force a tie; however, remodeling the
    # Gold into a Province wins outright.

    for player in state.players:
        player.n_turns_played = 5

    opponents[0].HAND.append(Duchy())

    state.need_action_phase = True
    state.n_action = 1
    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    copper = Copper()
    gold = Gold()
    remodel = Remodel()
    actor.HAND.append(copper)
    actor.HAND.append(gold)

    # Empty the final two piles.
    state.piles[-1].clear()
    state.piles[-2].clear()
    state.n_empty_piles = 2

    # Leave 1 piece in every non-empty pile.
    new_piles = []
    for pile in state.piles:
        new_pile = []
        if pile:
            topcard = pile[-1]
            new_pile = [topcard]
        new_piles.append(new_pile)
    state.piles = new_piles

    assert state.almost_over

    decision = remodel.decision
    choices = expand_choices(state, actor, decision)

    # All of the choices should be DependentAcquisitions.
    passes, nulls, purchases, dep_acqs, acqs, other = classify_consequences(choices)
    assert not(passes) and not(nulls) and not(purchases) and not(acqs) and not(other)
    assert all(isinstance(choice, DependentAcquisition) for choice in choices)

    pass_choice = None

    # The default selection method in these cases is select_dependent_acquisition.
    selection_method = actor.select_dependent_acquisition

    result = actor.meta_select(acquisitions=dep_acqs,
                               pass_choice=pass_choice,
                               selection_method=selection_method)

    assert state.acquisition_ends_game(result)
    assert would_win_outright(actor, result)
    assert isinstance(result.gained_piece, Province)

    # Demonstrate that this was the only option which accomplished
    # an outright win, but there was at least one option which
    # lost and one which tied.
    ties, losses = [], []
    for dependent_acquisition in dep_acqs:
        if (dependent_acquisition is not result):
            if state.acquisition_ends_game(dependent_acquisition):
                if not(would_win_outright(actor, dependent_acquisition)):
                    if (would_win(actor, dependent_acquisition)):
                        ties.append(dependent_acquisition)
                    else:
                        losses.append(dependent_acquisition)
    assert ties and losses


def test_recombine_sigma():
    """
    Ensure that the procedure for recombining permutations
    returns legal permutations.
    """
    N = 17
    reference = np.arange(N)
    top = np.arange(N)
    bot = np.arange(N)
    np.random.shuffle(top)
    np.random.shuffle(bot)
    recombined_result = recombine_sigma(top=top, bot=bot, N=17)
    return np.all(np.sort(recombined_result) == reference)


def test_conservation_of_n_genomes():
    """
    Ensure that the number of strategies replaced by the
    evolutionary algorithm doesn't vary.
    """
    simulation = Simulation(simname="test", N=8)
    simulation.random_spawn()
    original_n = len(simulation.genomes)
    simulation.evolve(n_generation=1, debug=True)
    final_n = len(simulation.genomes)
    assert original_n == final_n


def test_chapel_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid

    chapel = Chapel()

    # Case: Since Chapel is optional, having nothing
    # to trash should return a single option, to do nothing.
    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    choices = expand_choices(state, actor, chapel.decision)
    assert len(choices) == 1
    assert isinstance(choices[0], NullOption)

    # Case: Something to trash.
    actor.HAND.extend(actor.DISCARD)
    actor.DISCARD.clear()
    choices = expand_choices(state, actor, chapel.decision)
    assert len(choices) > 1
    assert isinstance(choices[0], NullOption)
    assert all((c.effects[0].function.__name__ == "trash_pieces") for c in choices[1:])

    # Choose one.
    choice = choices[-1]
    pieces_to_trash = choice.effects[0].kwargs['pieces']

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_trash = zone_data_t0['TRASH']
    for piece in pieces_to_trash:
        transfer_piece(piece, destination=t0_trash, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_harbinger_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    harbinger = Harbinger()
    predicted_changes = {
        'n_action':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, harbinger.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    transfer_top_piece(destination=t0_hand, source=t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)



def test_harbinger_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    harbinger = Harbinger()

    # Case: Nothing in discard.
    actor.DISCARD.clear()
    choices = expand_choices(state, actor, harbinger.decision)
    assert len(choices) == 1
    assert isinstance(choices[0], NullOption)

    # Case: Something to put on top.
    actor.DISCARD.append(harbinger)
    choices = expand_choices(state, actor, harbinger.decision)
    assert len(choices) == 2
    assert isinstance(choices[0], NullOption)

    choice = choices[1]
    effects = choice.effects
    assert effects[0].function.__name__ == "topdeck"


    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    transfer_top_piece(destination=t0_deck, source=t0_discard)

    validate_zone_data(zone_data_t1, zone_data_t0)


def test_vassal_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    vassal = Vassal()
    predicted_changes = {
        'n_coin':2,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, vassal.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_vassal_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    vassal = Vassal()

    # Case: Nothing in deck nor discard.
    actor.ASIDE.extend(actor.DISCARD)
    actor.ASIDE.extend(actor.DECK)
    actor.DISCARD.clear()
    actor.DECK.clear()

    choices = expand_choices(state, actor, vassal.decision)
    validate_null_option(choices)

    # Case: Single card in Deck, non-action.
    # Only choice is to Discard it.
    copper = Copper()
    actor.DECK.append(copper)

    choices = expand_choices(state, actor, vassal.decision)
    assert len(choices) == 1
    choice = choices[0]
    effects = choice.effects
    assert effects[0].function.__name__ == "discard_piece"

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    transfer_top_piece(destination=t0_discard, source=t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Case: Single card in Deck, action.
    # Either discard it, or play it.
    actor.DECK.append(vassal)

    choices = expand_choices(state, actor, vassal.decision)
    assert len(choices) == 2
    discard_choice = choices[0]
    discard_effects = discard_choice.effects
    assert discard_effects[0].function.__name__ == "discard_piece"
    play_choice = choices[1]
    play_effects = play_choice.effects
    assert play_effects[0].function.__name__ == "play_piece"

    # The simple effects of the Vassal will add 2 coin.
    # Playing it for free means n_action won't change.
    predicted_changes = {
        'n_coin':2
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, play_choice, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    t0_play = zone_data_t0[actor_pid]['PLAY']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    transfer_top_piece(destination=t0_play, source=t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_bureaucrat_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents
    opponent_pids = [opp.pid for opp in opponents]

    bureaucrat = Bureaucrat()

    # opponents[0] has a Moat so won't be affected.
    opponents[0].HAND.append(Moat())

    # opponents[1] has no victory cards in hand, so
    # should reveal their hand.
    opponents[1].DISCARD.extend(opponents[1].HAND)
    opponents[1].HAND.clear()
    opponents[1].HAND.append(Copper())

    # opponents[2] will need to put a victory card
    # on their deck.
    opponents[2].DISCARD.extend(opponents[2].HAND)
    opponents[2].HAND.clear()
    opponents[2].HAND.append(Duchy())

    choices = expand_choices(state, actor, bureaucrat.decision)
    assert len(choices) == 1
    choice = choices[0]

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect the predicted changes.
    # 1) Actor should have gained a Silver onto their Deck.
    silver_index = 5
    t0_deck = zone_data_t0[actor_pid]['DECK']
    t0_silver_pile = zone_data_t0['supply'][silver_index]
    transfer_top_piece(destination=t0_deck, source=t0_silver_pile)

    # 2) Opponents[2] should have topdecked their Duchy.
    t0_opp2_deck = zone_data_t0[opponent_pids[2]]['DECK']
    t0_opp2_hand = zone_data_t0[opponent_pids[2]]['HAND']
    transfer_top_piece(destination=t0_opp2_deck, source=t0_opp2_hand)

    validate_zone_data(zone_data_t1, zone_data_t0)


def test_gardens():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid

    gardens = Gardens()
    assert not(gardens.solve_points(actor.ASIDE))

    n_cards = len(actor.collection)
    n_cards_after_gardens = n_cards + 1
    gardens_vp_add = n_cards_after_gardens // 10
    base_vp = actor.victory_points
    predicted_vp = base_vp + gardens_vp_add

    actor.HAND.append(gardens)
    assert actor.victory_points == predicted_vp


def test_moneylender_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    moneylender = Moneylender()

    # Case: Nothing to Trash.
    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    choices = expand_choices(state, actor, moneylender.decision)
    validate_null_option(choices)

    # Case: Copper to Trash.
    actor.HAND.append(Copper())
    choices = expand_choices(state, actor, moneylender.decision)
    # Optional effect implies two choices: Do nothing, and trash it.
    assert len(choices) == 2
    assert isinstance(choices[0], NullOption)

    choice = choices[1]
    effects = choice.effects
    assert effects[0].function.__name__ == "trash"
    assert effects[1].function.__name__ == "add_coin"

    predicted_changes = {
        'n_coin':3
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_trash = zone_data_t0['TRASH']
    transfer_top_piece(destination=t0_trash, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_poacher_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    poacher = Poacher()

    predicted_changes = {
        'n_action':1,
        'n_coin':1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, poacher.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    transfer_top_piece(destination=t0_hand, source=t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_poacher_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    poacher = Poacher()

    # Case: No empty Supply piles.
    choices = expand_choices(state, actor, poacher.decision)
    validate_null_option(choices)

    # Case: Two empty supply piles.
    state.piles[0].clear()
    state.piles[1].clear()
    state.n_empty_piles = 2

    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()
    actor.HAND.append(Gold())
    actor.DECK.append(Gold())

    choices = expand_choices(state, actor, poacher.decision)
    choice = choices[0]

    predicted_changes = {
        'n_action':1,
        'n_coin':1,
    }

    # Test together with simple effects since adding a card
    # influences discard decisions.
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, poacher.simple_effects, **predicted_changes)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # Alter zone_data_t0 to reflect predicted changes.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_deck = zone_data_t0[actor_pid]['DECK']
    t0_discard = zone_data_t0[actor_pid]['DISCARD']
    transfer_top_piece(destination=t0_hand, source=t0_deck)
    transfer_top_piece(destination=t0_discard, source=t0_hand)
    transfer_top_piece(destination=t0_discard, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_throneroom_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    tr0 = ThroneRoom()
    tr1 = ThroneRoom()
    vassal = Vassal()

    actor.DISCARD.extend(actor.HAND)
    actor.HAND.clear()

    # Case: No actions to double play.
    choices = expand_choices(state, actor, tr0.decision)
    validate_null_option(choices)

    # Case: A Throne Room in hand to double play.
    actor.HAND.append(tr1)
    choices = expand_choices(state, actor, tr0.decision)
    assert len(choices) == 2
    assert isinstance(choices[0], NullOption)

    choice = choices[1]
    effects = choice.effects
    assert len(effects) == 2
    assert effects[0].function.__name__ == "play_piece"
    assert effects[1].function.__name__ == "resolve_effects"

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_play = zone_data_t0[actor_pid]['PLAY']
    transfer_top_piece(destination=t0_play, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Case: A Vassal in Hand to double play.
    actor.HAND.append(vassal)

    # Simplify things for the Vassal.
    actor.ASIDE.extend(actor.DECK)
    actor.ASIDE.extend(actor.DISCARD)
    actor.DECK.clear()
    actor.DISCARD.clear()

    predicted_changes = {
        'n_coin':4,
    }

    choices = expand_choices(state, actor, tr0.decision)
    choice = choices[1]
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_play = zone_data_t0[actor_pid]['PLAY']
    transfer_top_piece(destination=t0_play, source=t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_bandit_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents
    opponent_pids = [opp.pid for opp in opponents]

    bandit = Bandit()

    # opponents[0] has a Moat, so won't be affected.
    opponents[0].HAND.append(Moat())

    # opponents[1] has only a single card in Deck
    # and Discard combined, and it is a copper.
    opponents[1].ASIDE.extend(opponents[1].DECK)
    opponents[1].ASIDE.extend(opponents[1].DISCARD)
    opponents[1].DECK.clear()
    opponents[1].DISCARD.clear()
    opponents[1].DECK.append(Copper())

    # opponents[2] has a Silver and a Vassal as
    # the top two cards.
    opponents[2].DECK.append(Silver())
    opponents[2].DECK.append(Vassal())

    choices = expand_choices(state, actor, bandit.decision)
    assert len(choices) == 1
    choice = choices[0]
    # Gold Pile isn't empty, so should be gaining a Gold.
    assert isinstance(choice, Acquisition)
    gold_index = 6

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # Attacker gains a Gold.
    t0_hand = zone_data_t0[actor_pid]['HAND']
    t0_gold = zone_data_t0['supply'][gold_index]
    transfer_top_piece(destination=t0_hand, source=t0_gold)

    # opponents[1]'s top card was discarded.
    t1_deck = zone_data_t0[opponent_pids[1]]['DECK']
    t1_discard = zone_data_t0[opponent_pids[1]]['DISCARD']
    transfer_top_piece(destination=t1_discard, source=t1_deck)

    # opponents[2] top card was discarded and second top card
    # was trashed.
    t2_deck = zone_data_t0[opponent_pids[2]]['DECK']
    t2_discard = zone_data_t0[opponent_pids[2]]['DISCARD']
    t2_trash = zone_data_t0['TRASH']
    transfer_top_piece(destination=t2_discard, source=t2_deck)
    transfer_top_piece(destination=t2_trash, source=t2_deck)

    validate_zone_data(zone_data_t1, zone_data_t0)


def test_councilroom_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    councilroom = CouncilRoom()

    predicted_changes = {
        'n_buy':1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, councilroom.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    valid_draw(pid=actor.pid, n=4, zd_t0=zone_data_t0)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_councilroom_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    opponents = actor.opponents
    opponent_pids = [opp.pid for opp in opponents]
    councilroom = CouncilRoom()

    predicted_changes = {
        'n_buy':1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, councilroom.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    valid_draw(pid=actor.pid, n=4, zd_t0=zone_data_t0)
    for opponent_pid in opponent_pids:
        valid_draw(pid=opponent_pid, n=1, zd_t0=zone_data_t0)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_festival_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    festival = Festival()
    predicted_changes = {
        'n_action':2,
        'n_buy':1,
        'n_coin':2,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, festival.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_laboratory_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    laboratory = Laboratory()

    predicted_changes = {
        'n_action':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, laboratory.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    valid_draw(pid=actor.pid, n=2, zd_t0=zone_data_t0)
    validate_zone_data(zone_data_t1, zone_data_t0)



def test_library_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    library = Library()

    # Remove actor's Deck and replace it with two actions.
    # Remove actor's Discard for convenience.
    actor.DECK.clear()
    actor.DISCARD.clear()
    actor.DECK.append(Cellar())
    actor.DECK.append(Vassal())

    choices = expand_choices(state, actor, library.decision)
    assert len(choices) == 1
    choice = choices[0]

    assert len(actor.HAND) == 5

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)

    # The choice to set the Actions aside will be done
    # at random. Instead of fixing a random seed, consider
    # the cases.

    t0_deck = zone_data_t0[actor.pid]['DECK']
    t0_hand = zone_data_t0[actor.pid]['HAND']
    t0_discard = zone_data_t0[actor.pid]['DISCARD']

    # Case: There are 2 cards in actor.DISCARD, implying
    #       they set aside both Actions.
    if (len(actor.DISCARD) == 2):
        transfer_top_piece(destination=t0_discard, source=t0_deck)
        transfer_top_piece(destination=t0_discard, source=t0_deck)

    # Case: There are 0 cards in actor.DISCARD, implying
    #       they chose to keep both Actions.
    elif not(actor.DISCARD):
        transfer_top_piece(destination=t0_hand, source=t0_deck)
        transfer_top_piece(destination=t0_hand, source=t0_deck)

    # Case: There is 1 card in actor.DISCARD, implying they
    #       skipped one of the Actions.
    elif (len(actor.DISCARD) == 1):
        # Case: They drew Vassal, then skipped Cellar.
        if isinstance(actor.DISCARD[-1], Cellar):
            transfer_top_piece(destination=t0_hand, source=t0_deck)
            transfer_top_piece(destination=t0_discard, source=t0_deck)

        # Case: They skipped Vassal, then drew Cellar.
        elif isinstance(actor.DISCARD[-1], Vassal):
            transfer_top_piece(destination=t0_discard, source=t0_deck)
            transfer_top_piece(destination=t0_hand, source=t0_deck)

    validate_zone_data(zone_data_t1, zone_data_t0)


def test_sentry_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    sentry = Sentry()

    predicted_changes = {
        'n_action':1,
    }
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, sentry.simple_effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)
    valid_draw(pid=actor.pid, n=1, zd_t0=zone_data_t0)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_sentry_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    sentry = Sentry()

    # Case: Nothing in Deck nor Discard.
    actor.ASIDE.extend(actor.DISCARD)
    actor.ASIDE.extend(actor.DECK)
    actor.DISCARD.clear()
    actor.DECK.clear()

    choices = expand_choices(state, actor, sentry.decision)
    validate_null_option(choices)

    # Case: Single card in Deck. Note, not combining with
    #       the simple effects, which would cause the
    #       single card to be drawn before the choices are
    #       generated.
    actor.DECK.append(Silver())
    choices = expand_choices(state, actor, sentry.decision)

    # NullOption means leaving it on top.
    assert isinstance(choices[0], NullOption)
    trash_choice = choices[1]
    discard_choice = choices[2]

    # Trashing it.
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, trash_choice.effects)
    zone_data_t1 = extract_zone_data(state)

    t0_trash = zone_data_t0['TRASH']
    t0_deck = zone_data_t0[actor.pid]['DECK']
    transfer_top_piece(t0_trash, t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Undo that.
    transfer_top_piece(actor.DECK, state.TRASH)

    # Discarding it.
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, discard_choice.effects)
    zone_data_t1 = extract_zone_data(state)

    t0_discard = zone_data_t0[actor.pid]['DISCARD']
    t0_deck = zone_data_t0[actor.pid]['DECK']
    transfer_top_piece(t0_discard, t0_deck)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Undo that.
    transfer_top_piece(actor.DECK, actor.DISCARD)

    # Case: Two cards which differ in identity in Deck.
    actor.DECK.append(Gold())

    choices = expand_choices(state, actor, sentry.decision)
    # First choice represents leaving them in the same order.
    assert isinstance(choices[0], NullOption)

    # Test swapping their positions.
    swap_choice = choices[3]
    effects = swap_choice.effects[0]
    assert effects.function.__name__ == "swap_top_cards_of_deck"

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, swap_choice.effects)
    zone_data_t1 = extract_zone_data(state)

    t0_deck = zone_data_t0[actor.pid]['DECK']
    card0 = t0_deck.pop()
    card1 = t0_deck.pop()
    t0_deck.append(card1)
    t0_deck.append(card0)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_witch_simple_effects():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    witch = Witch()
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, witch.simple_effects)
    zone_data_t1 = extract_zone_data(state)
    valid_draw(pid=actor.pid, n=2, zd_t0=zone_data_t0)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_witch_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    opponents = actor.opponents
    opponent_pids = [opponent.pid for opponent in opponents]
    witch = Witch()

    # opponents[0] has a Moat so won't be affected.
    opponents[0].HAND.append(Moat())

    # In this contrived example, there will only be
    # one Curse remaining in the Supply; thus the first
    # non-immune opponent will end up emptying it and
    # the second non-immune opponent won't gain anything.
    curse_index = 0
    state.piles[curse_index] = [Curse()]
    choices = expand_choices(state, actor, witch.decision)
    assert len(choices) == 1
    choice = choices[0]

    predicted_changes = {
        'n_empty_piles':1,
    }

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects, **predicted_changes)
    zone_data_t1 = extract_zone_data(state)

    t0_curse = zone_data_t0['supply'][curse_index]
    t0_opp1_discard = zone_data_t0[opponent_pids[1]]['DISCARD']
    transfer_top_piece(t0_opp1_discard, t0_curse)
    validate_zone_data(zone_data_t1, zone_data_t0)


def test_artisan_choices():
    S = get_test_session()
    S.start_turn()
    state = S.state
    actor = state.current_player
    actor_pid = actor.pid
    artisan = Artisan()

    # Case: Pathological case where there is nothing to gain.
    temp_supply = [list(pile) for pile in state.piles]
    for pile in state.piles:
        pile.clear()

    # Nothing in hand, either.
    actor.ASIDE.extend(actor.HAND)
    actor.HAND.clear()

    choices = expand_choices(state, actor, artisan.decision)
    validate_null_option(choices)

    # One card in hand; only option is to gain nothing and
    # topdeck it.
    actor.HAND.append(Silver())
    choices = expand_choices(state, actor, artisan.decision)
    choice = choices[0]

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)
    t0_hand = zone_data_t0[actor.pid]['HAND']
    t0_deck = zone_data_t0[actor.pid]['DECK']
    transfer_top_piece(t0_deck, t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)

    silver_index = 5
    # Nothing in hand, but something worth gaining in
    # the Supply; adding multiple to prevent triggering
    # empty pile detection which is irrelevant in this
    # contrived example.
    state.piles[silver_index] = [Silver(), Silver(), Silver()]

    choices = expand_choices(state, actor, artisan.decision)
    # Only choice is to gain that piece and immediately
    # top deck it.
    choice = choices[0]

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, choice.effects)
    zone_data_t1 = extract_zone_data(state)
    t0_hand = zone_data_t0[actor.pid]['HAND']
    t0_deck = zone_data_t0[actor.pid]['DECK']
    t0_silver = zone_data_t0['supply'][silver_index]
    transfer_top_piece(t0_hand, t0_silver)
    transfer_top_piece(t0_deck, t0_hand)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Something in hand with the same identity as
    # the gainable Piece; only only choice should
    # be generated.
    actor.HAND.append(Silver())
    choices = expand_choices(state, actor, artisan.decision)
    assert len(choices) == 1

    # Something in hand with a different identity
    # than the gainable Piece; two choices should
    # be generated.
    actor.ASIDE.extend(actor.HAND)
    actor.HAND.clear()
    actor.HAND.append(Gold())
    choices = expand_choices(state, actor, artisan.decision)
    assert len(choices) == 2

    # First choice will be to gain the Silver and
    # topdeck the card already in hand.
    topdeck_gold = choices[0]

    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, topdeck_gold.effects)
    zone_data_t1 = extract_zone_data(state)
    t0_hand = zone_data_t0[actor.pid]['HAND']
    t0_deck = zone_data_t0[actor.pid]['DECK']
    t0_silver = zone_data_t0['supply'][silver_index]
    transfer_top_piece(t0_deck, t0_hand)
    transfer_top_piece(t0_hand, t0_silver)
    validate_zone_data(zone_data_t1, zone_data_t0)

    # Undo the gaining.
    transfer_top_piece(state.piles[silver_index], actor.HAND)
    #Undo the topdecking. 
    transfer_top_piece(actor.HAND, actor.DECK)

    # Second choice will be to gain the Silver and
    # topdeck it.
    topdeck_silver = choices[1]
    zone_data_t0 = extract_zone_data(state)
    validate_effects(state, topdeck_gold.effects)
    zone_data_t1 = extract_zone_data(state)
    t0_hand = zone_data_t0[actor.pid]['HAND']
    t0_deck = zone_data_t0[actor.pid]['DECK']
    t0_silver = zone_data_t0['supply'][silver_index]
    transfer_top_piece(t0_deck, t0_silver)
    validate_zone_data(zone_data_t1, zone_data_t0)
