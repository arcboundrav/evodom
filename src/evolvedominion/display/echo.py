from evolvedominion.params import (
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
)
from evolvedominion.engine.session import Session
from evolvedominion.display.text import (
    announce_event,
    display_title,
    display_buffer_line,
)


class Echo:
    """
    Mixin to support configuring whether actions are echoed.
    Decouples text display from default execution so performance
    during simulations, which never display text, isn't compromised.
    """
    def select(self, choices, decision):
        consequence = super().select(choices, decision)
        announce_event(self, consequence)
        return consequence


class EchoSession(Session):
    """
    Extend Session with hooks to support text representation of the game
    for human players.
    """
    def start_turn(self):
        super().start_turn()
        if not(self.state.current_player_index):
            display_title("Turn {}".format(self.state.current_player.n_turns_played))

    def action_phase(self):
        display_title("{}".format(ACTION_PHASE))
        super().action_phase()

    def treasure_phase(self):
        super().treasure_phase()
        display_buffer_line()

    def buy_phase(self):
        display_title("{}".format(BUY_PHASE))
        super().buy_phase()

    def end_turn(self):
        super().end_turn()
        display_buffer_line()
