import os
import re
import click

from evolvedominion.params import PHASES
from evolvedominion.display.conf import VERBOSE_OPPONENTS


CARD_MAP = {
    "Curse":{"cost":"0", "points":"-1", "value":"0"},
    "Estate":{"cost":"2", "points":"1", "value":"0"},
    "Duchy":{"cost":"5", "points":"3", "value":"0"},
    "Province":{"cost":"8", "points":"6", "value":"0"},
    "Copper":{"cost":"0", "points":"0", "value":"1"},
    "Silver":{"cost":"3", "points":"0", "value":"2"},
    "Gold":{"cost":"6", "points":"0", "value":"3"},
    "Cellar":{"cost":"2", "points":"0", "value":"0"},
    "Moat":{"cost":"2", "points":"0", "value":"0"},
    "Merchant":{"cost":"3", "points":"0", "value":"0"},
    "Workshop":{"cost":"3", "points":"0", "value":"0"},
    "Village":{"cost":"3", "points":"0", "value":"0"},
    "Smithy":{"cost":"4", "points":"0", "value":"0"},
    "Remodel":{"cost":"4", "points":"0", "value":"0"},
    "Militia":{"cost":"4", "points":"0", "value":"0"},
    "Market":{"cost":"5", "points":"0", "value":"0"},
    "Mine":{"cost":"5", "points":"0", "value":"0"},
}

CARD_NAMES = list(CARD_MAP.keys())

LOWER_CARD_NAMES = [str.lower(card_name) for card_name in CARD_NAMES]

CARD_DETAILS = {
    "Cellar":[
        "+1 Action",
        "Discard any number of Victory Cards, then draw that many cards."
    ],
    "Market":[
        "+1 Card",
        "+1 Action",
        "+1 Buy",
        "+1 Coin"
    ],
    "Merchant":[
        "+1 Card",
        "+1 Action",
        "The first time you play a Silver this turn, +1 Coin."
    ],
    "Militia":[
        "+2 Coin",
        "Each other player discards down to three cards in hand."
    ],
    "Mine":[
        "You may trash a Treasure from your hand.",
        "Gain a Treasure to your hand costing up to 3 more than it."
    ],
    "Moat":[
        "+2 Cards",
        "When another player plays Militia, reveal this card to be unaffected by it."
    ],
    "Remodel":[
        "Trash a card from your hand.",
        "Gain a card costing up to 2 more than it."
    ],
    "Smithy":[
        "+3 Cards"
    ],
    "Village":[
        "+1 Card",
        "+2 Actions"
    ],
    "Workshop":[
        "Gain a card costing up to 4."
    ],
}

L = str.ljust

R = str.rjust

C = str.center

TAB = "    "

VIEW_PATTERN = r'view (?P<name>[A-Za-z]+)$'

BAD_VIEW_PATTERN = r'view( )?$'

TOP_PROMPT = ''.join(["|", " ".join(["Coins", "Actions", "Buys"]), "|\n"])

BOTTOM_PROMPT = '|{} {} {}|> '

SEEK_KEYPRESS_PROMPT = "---Press any key to continue [{}/{}]---"

VERBS = {
    'end_phase':["end the", "ends the"],
    'buy_piece':["buy", "buys"],
    'gain':["gain", "gains"],
    'play_piece':["play", "plays"],
    'play_treasures':["play", "plays"],
    'discard':["discard", "discards"],
    'discard_pieces':["discard", "discards"],
    'discard_zone':["discard", "discards"],
    'reveal':["reveal", "reveals"],
    'trash':["trash", "trashes"],
    'trash_pieces':["trash", "trashes"],
    'draw':["draw", "draws"],
    'do_nothing':["do nothing", "does nothing"],
    'topdeck':["topdeck", "topdecks"],
    'put':["put", "puts"],
    'set_aside':["set aside", "sets aside"],
    'swap_top_two_cards':["swap the top two cards", "swaps the top two cards"],
    'resolve_effects':["replay", "replays"],
    'resolve':["resolve", "resolves"],
}

COMMAND_HINTS = {
    "?":"Display command hints.",
    "h":"Display the cards in your Hand.",
    "p":"Display the cards you have in Play.",
    "d":"Display the top card of your Discard pile.",
    "k":"Display the number of cards in your Deck.",
    "s":"Display the Supply.",
    "t":"Display the Trash.",
    "o":"Display the available options to choose from.",
    "i":"Display information about each Player.",
    "view <card>":"View card details. Example: view copper"
}

ZONE_TITLES = {
    "DISCARD":"Top of Your Discard Pile",
    "HAND":"Your Hand",
    "PLAY":"Cards in Your Play Area",
    "TRASH":"Trashed Cards"
}


def uppercase(string):
    """
    Capitalize the first character in string without
    altering the case of any other characters.
    """
    head, tail = string[0], string[1:]
    return "{}{}".format(head.capitalize(), tail)


def strict_uppercase(string):
    """
    Ensure only the first character in string is
    capitalized, with the remainder lower cased.
    """
    return uppercase(str.lower(string))


def _solve_buffer_line():
    """ Produce a horizontal line matching the terminal's width. """
    return "_" * os.get_terminal_size().columns


def display_buffer_line():
    print(_solve_buffer_line())


def buffer_line(function):
    """
    Decorator for display functions which should be
    sandwiched between two horizontal lines.
    """
    def f(*args, **kwargs):
        buffer_line = _solve_buffer_line()
        print(buffer_line)
        function(*args, **kwargs)
        print(buffer_line)
    return f


def print_list(list_of_strings):
    for string in list_of_strings:
        print(string)


def display_title(title_string):
    """ Print title_string in the center of the terminal. """
    if title_string:
        width = os.get_terminal_size().columns
        print(title_string.center(width))


def auto_print(list_of_strings, gap_size=1, justify=L):
    """
    Solve the maximum number of columns the contents of
    list_of_strings can be displayed in given the gap_size,
    then display the strings in those columns. Paginate
    the results if the implied number of rows exceeds the
    terminal size.
    """
    assert isinstance(list_of_strings, list)
    assert list_of_strings and all(list_of_strings)
    _print_columns(list_of_strings, gap_size=gap_size, justify=justify)


def _solve_column_width(list_of_strings):
    """ Return the length of the longest string in list_of_strings """
    return len(sorted(list_of_strings, key=lambda string: len(string), reverse=True)[0])


def _compute_n_columns(list_of_strings, width, col_width, gap_size):
    """ Return the maximum number of columns given the constraints. """
    n_cols, n_residual = divmod(width, col_width)
    if (n_cols > 1):
        n_gaps = n_cols - 1
        n_padding = n_gaps * gap_size
        while (n_padding > n_residual):
            n_cols = n_cols - 1
            n_gaps = n_cols - 1
            n_padding = n_gaps * gap_size
            n_residual = n_residual + col_width
    return n_cols


def _print_columns(list_of_strings, gap_size, justify):
    width = os.get_terminal_size().columns
    col_width = _solve_column_width(list_of_strings)
    if (col_width > width):
        print_list(list_of_strings)
    else:
        n_cols = _compute_n_columns(list_of_strings, width, col_width, gap_size)
        strings = [justify(string, col_width) for string in list_of_strings]
        if (n_cols < 2):
            _print_rows(strings)
        else:
            lines = []
            gap = " " * gap_size
            n_strings = len(strings)
            for i in range(0, n_strings, n_cols):
                lines.append(gap.join(strings[i:(i + n_cols)]))
            _print_rows(lines)


def _print_rows(lines):
    n_rows = len(lines)
    height = os.get_terminal_size().lines
    # Case: When necessary and if able, present the rows
    # without vertical scrolling.
    if ((n_rows > height) and (height > 1)):
        # NOTE # Leave a row to display the user input prompt.
        n_rows_per_block = height - 1
        n_blocks = 0
        blocks = []
        for i in range(0, n_rows, n_rows_per_block):
            blocks.append(lines[i:(i + n_rows_per_block)])
            n_blocks = n_blocks + 1
        print_blocks(blocks=blocks, n_blocks=n_blocks)
    else:
        print_list(lines)


def seek_keypress(block_index, n_blocks):
    click.pause(info=SEEK_KEYPRESS_PROMPT.format(block_index, n_blocks))


def print_blocks(blocks, n_blocks):
    block_index = 0
    while (block_index < n_blocks):
        block = blocks[block_index]
        block_index = block_index + 1
        print_list(block)
        # NOTE # Only request user input to advance to the next block when needed.
        if (block_index != n_blocks):
            seek_keypress(block_index, n_blocks)


def _find_column_maxes(rows, n_columns):
    column_maxes = [0 for i in range(len(rows[0]))]
    for row in rows:
        for i, entry in enumerate(row):
            if (len(entry) > column_maxes[i]):
                column_maxes[i] = len(entry)
    return column_maxes


def _create_layout_rows(rows, column_maxes, width, gap, justify, center_rows):
    layout_rows = []
    for row in rows:
        for i, column_max in enumerate(column_maxes):
            row[i] = justify[i](row[i], column_max)
        row = C(gap.join(row), width) if center_rows else gap.join(row)
        layout_rows.append(row)
    return layout_rows


def layout(title, rows, gap, justify, center_rows=True):
    width = os.get_terminal_size().columns
    n_columns = len(rows[0])
    if (len(justify) < n_columns):
        justify = [justify[0]] * n_columns
    column_maxes = _find_column_maxes(rows, n_columns)
    rows_to_print = _create_layout_rows(rows, column_maxes, width, gap, justify, center_rows)
    display_title(title)
    _print_rows(rows_to_print)


def newline():
    print("\n")


def with_newlines(list_of_strings):
    return "\n".join(list_of_strings)


def with_spaces(list_of_strings, n=1):
    gap = " " * n
    return gap.join(list_of_strings)


def add_period(string):
    return "{}.".format(string)


def represent_pieces(pieces):
    """ Expects a tuple or list of Piece instances. """
    n_pieces = len(pieces)
    if (n_pieces > 2):
        result = ", ".join([repr(piece) for piece in pieces[:-1]])
        return "{}, and {}".format(result, repr(pieces[-1]))
    elif (n_pieces == 2):
        return " and ".join([repr(piece) for piece in pieces])
    return repr(pieces[0])


def summarize_actor(actor, endgame=False):
    if endgame:
        count = str(len(actor.collection))
    else:
        count = str(len(actor.HAND))
    return [repr(actor), count, str(actor.victory_points), str(actor.n_turns_played)]


@buffer_line
def summarize_session(session):
    title = "Session Summary"
    rows = [['', '', "# of Cards", "VP", "Turns Played"]]
    place = 1
    for i, player in enumerate(session.players):
        if i:
            j = i - 1
            other = session.players[j]
            if (player < other):
                place = place + 1
        row = [str(place)]
        row.extend(summarize_actor(player, endgame=True))
        rows.append(row)
    layout(title=title, rows=rows, gap=TAB, justify=[L, R, C, R, C])


def represent_state_coin(state):
    return str(state.n_coin).ljust(5)


def represent_state_n_action(state):
    return str(state.n_action).ljust(7)


def represent_state_n_buy(state):
    return str(state.n_buy).ljust(4)

def represent_state_phase(state):
    if state.need_action_phase:
        return "[Action Phase]"
    return "[Buy Phase]"

def solve_prompt(state):
    bottom_line = BOTTOM_PROMPT.format(
        represent_state_coin(state),
        represent_state_n_action(state),
        represent_state_n_buy(state)
    )
    return "{}{}".format(TOP_PROMPT, bottom_line)


def _pile_row(pile):
    top_card = pile[-1]
    return [top_card.__repr__(), str(top_card.cost), str(top_card.points), str(len(pile))]


@buffer_line
def display_supply(**kwargs):
    state = kwargs['state']
    piles_to_display = sorted([pile for pile in state.piles if pile], key=lambda pile: (pile[-1].cost))
    rows = [["", "Cost", "VP", "# Remaining"]]
    rows.extend([_pile_row(pile) for pile in piles_to_display])
    layout(title="The Supply",
           rows=rows,
           gap=TAB,
           justify=[L, C, R, C])


def commas(list_of_pieces):
    return ', '.join("{}".format(piece) for piece in list_of_pieces)


def represent_effect_kwargs(kwargs):
    result = []
    for kwarg in kwargs:
        if (kwarg != "actor"):
            if (kwarg == "pieces"):
                result.append(represent_pieces(kwargs[kwarg]))
            elif (kwarg not in ["source", "destination", "state", "is_buy"]):
                result.append(kwargs[kwarg])
    return commas(result)


def represent_effect(effect, idx=0):
    verb_string = '{}'.format(VERBS[effect.function.__name__][idx])
    kwargs_string = represent_effect_kwargs(effect.kwargs)
    if kwargs_string:
        return "{} {}".format(verb_string, kwargs_string)
    return verb_string


def represent_effects(effects, lowercase, idx):
    effects_to_represent = list(filter(lambda effect: not(effect.__class__.__name__ == "Update"), effects))
    effects = '; then '.join([represent_effect(effect, idx) for effect in effects])
    effects = effects if lowercase else uppercase(effects)
    return add_period(effects)


def represent_consequence(consequence, lowercase=False, idx=0):
    """
    First-person perspective when idx == 0.
    Third-person perspective when idx == 1.
    """
    if consequence.custom_message:
        return consequence.custom_message[idx]
    return represent_effects(consequence.effects, lowercase, idx)


@buffer_line
def _display_zone(ZONE_STRING, zone_owner):
    zone = getattr(zone_owner, ZONE_STRING)
    list_of_strings = [repr(piece) for piece in zone]
    display_title(ZONE_TITLES[ZONE_STRING])
    if list_of_strings:
        if (ZONE_STRING == "DISCARD"):
            list_of_strings = list_of_strings[-1:]
        auto_print(list_of_strings, 4)


@buffer_line
def display_deck(**kwargs):
    zone_owner = kwargs['actor']
    zone = getattr(zone_owner, 'DECK')
    n = len(zone)
    display_title("Number of Cards in Your Deck: {}".format(n))


def display_hand(**kwargs):
    return _display_zone(ZONE_STRING="HAND", zone_owner=kwargs['actor'])


def display_play(**kwargs):
    return _display_zone(ZONE_STRING="PLAY", zone_owner=kwargs['actor'])


def display_discard(**kwargs):
    return _display_zone(ZONE_STRING="DISCARD", zone_owner=kwargs['actor'])


def display_trash(**kwargs):
    return _display_zone(ZONE_STRING="TRASH", zone_owner=kwargs['state'])


@buffer_line
def display_progress(**kwargs):
    title = "Player Information"
    header_row = ['', '', "Cards in Hand", "VP", "Turns Played"]
    rows = [header_row]
    state = kwargs['state']
    for player in state.players:
        row = [represent_state_phase(state)] if (player == state.current_player) else [""]
        row.extend(summarize_actor(player))
        rows.append(row)
    layout(title=title, rows=rows, gap=TAB, justify=[L, R, C, R, C])


@buffer_line
def display_choices(**kwargs):
    lines = []
    for i, choice in enumerate(kwargs['choices']):
        lines.append(["{}".format(str(i)), "|", "{}".format(represent_consequence(choice))])
    layout(title="",
           rows=lines,
           gap=" ",
           justify=[R, C, L])


@buffer_line
def display_input_hint():
    print("Please enter an option number; or, enter ? to display commands.")




@buffer_line
def display_commands(**kwargs):
    layout(title="COMMANDS",
           rows=[["{}".format(cmd), "{}".format(COMMAND_HINTS[cmd])] for cmd in COMMAND_HINTS],
           gap=TAB,
           justify=[L])


@buffer_line
def _view_hints(**kwargs):
    print("You may view the following cards:")
    auto_print(LOWER_CARD_NAMES, gap_size=4)


COMMAND_MAP = {
    "?":display_commands,
    "h":display_hand,
    "p":display_play,
    "d":display_discard,
    "k":display_deck,
    "s":display_supply,
    "t":display_trash,
    "o":display_choices,
    "i":display_progress,
}


@buffer_line
def _details(card_name):
    card_text = "" if not(card_name in CARD_DETAILS) else CARD_DETAILS[card_name]
    headers = ["Name", "Cost", "Victory Points", "Value"]
    details = [
        card_name,
        CARD_MAP[card_name]['cost'],
        CARD_MAP[card_name]['points'],
        CARD_MAP[card_name]['value']
    ]
    layout(title="",
           rows=[headers, details],
           gap=TAB,
           justify=[L, C, C, C],
           center_rows=False)
    if card_text:
        _print_rows(card_text)


def _view_parser(cmd):
    """
    Use regular expressions to detect attempts to view Piece details.
    """
    match = re.match(VIEW_PATTERN, cmd)
    if (match is not None):
        card_name = strict_uppercase(match.groupdict()['name'])
        if (card_name in CARD_MAP):
            _details(card_name)
            return True
        else:
            _view_hints()
            return True
    else:
        match = re.match(BAD_VIEW_PATTERN, cmd)
        if (match is not None):
            _view_hints()
            return True
    return False


def parse_display_command(cmd, **kwargs):
    """ Interpret user input and respond accordingly. """
    if not(_view_parser(cmd)):
        if (cmd in COMMAND_MAP):
            COMMAND_MAP[cmd](**kwargs)
        else:
            display_input_hint()


def announce_event(actor, consequence):
    # Announce events from a third person perspective by default.
    idx = 1

    # Case: Announce event from a first person perspective.
    if (repr(actor) == "You"):
        idx = 0

    # Always display the actions of human players.
    # For computer players,refer to configured display preferences.
    if (not(idx) or (VERBOSE_OPPONENTS and idx)):
        print("> {} {}".format(actor, represent_consequence(consequence, lowercase=True, idx=idx)))


def announce_drawn_cards(pieces, n_turns_played):
    if n_turns_played:
        print("> You drew {}.".format(represent_pieces(pieces)))



def display_generation_duration(generation_index, n_generation, t1, t0):
    duration = round((t1 - t0), 2)
    print("Generation: {} / {} [{} seconds]".format((generation_index + 1),
                                            n_generation,
                                            duration))

def display_simulation_duration(simname, t1, t0):
    duration = round((t1 - t0), 2)
    print("End of simulation {} reached after {} seconds.".format(simname, duration))

