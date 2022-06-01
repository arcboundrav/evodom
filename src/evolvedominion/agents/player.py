from evolvedominion.agents.agent import Agent
from evolvedominion.display.text import (
    display_choices,
    parse_display_command,
    display_buffer_line,
    solve_prompt,
    announce_drawn_cards,
)
from evolvedominion.display.echo import Echo


class Player(Agent):
    """
    Supports human players selecting options via text interface.
    QoL features: automatically choose single options, play
    the maximum number of Treasures during the Treasure Phase,
    and echo drawn cards.
    """
    def select(self, choices, decision):
        automatic_choice = self._automatic(choices, decision)
        if (automatic_choice is not None):
            return automatic_choice
        legal_indices = set(str(i) for i in range(len(choices)))
        response = None
        display_choices(actor=self, state=self.state, choices=choices)
        while (response not in legal_indices):
            response = input(solve_prompt(self.state))
            if not(response in legal_indices):
                parse_display_command(response, actor=self, state=self.state, choices=choices)
        display_buffer_line()
        return choices[int(response)]

    def peek(self, cards):
        announce_drawn_cards(cards, self.n_turns_played)

    def __repr__(self):
        return "You"


class EchoPlayer(Echo, Player):
    pass
