from evolvedominion.params import (
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
)
from evolvedominion.engine.session import Session
from evolvedominion.display.text import (
    announce_epoch_start,
    announce_epoch_end,
    announce_event,
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
        announce_epoch_start("Turn")
        super().start_turn()

    def action_phase(self):
        announce_epoch_start(ACTION_PHASE)
        super().action_phase()

    def treasure_phase(self):
        announce_epoch_start(TREASURE_PHASE)
        super().treasure_phase()

    def buy_phase(self):
        announce_epoch_start(BUY_PHASE)
        super().buy_phase()

    def end_turn(self):
        #announce_epoch_end("Turn")
        super().end_turn()
        display_buffer_line()
