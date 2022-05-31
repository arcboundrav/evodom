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
    def _start_turn(self):
        announce_epoch_start("Turn")
        super()._start_turn()

    def _action_phase(self):
        announce_epoch_start(ACTION_PHASE)
        super()._action_phase()

    def _treasure_phase(self):
        announce_epoch_start(TREASURE_PHASE)
        super()._treasure_phase()

    def _buy_phase(self):
        announce_epoch_start(BUY_PHASE)
        super()._buy_phase()

    def _end_turn(self):
        announce_epoch_end("Turn")
        super()._end_turn()
