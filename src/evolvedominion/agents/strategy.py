import numpy as np

from numba import njit
from itertools import tee, filterfalse
from evolvedominion.agents.agent import Agent
from evolvedominion.display.echo import Echo
from evolvedominion.types import SMALL_INT, BOOL
from evolvedominion.engine.combinatorics import (
    partition,
    classify,
)
from evolvedominion.engine.engine import (
    evaluate_acquisitions,
    extract_acquisitions,
    classify_acts,
    classify_consequences,
    expand_choices,
    NullOption,
    PassOption,
    Purchase,
    DependentAcquisition,
    Acquisition,
    ActionChoices,
    BuyChoices,
    CellarChoices,
    MineChoices,
    RemodelChoices,
    WorkshopChoices,
    TreasureChoices,
    MilitiaVictimProcedure,
)
from evolvedominion.params import (
    N_PURCHASE_PREFERENCES,
    N_PIECE_IDENTITIES,
    N_ACTION_CLASSES,
)


PIECE_IDENTITIES = np.arange(N_PIECE_IDENTITIES, dtype=SMALL_INT)
ACTION_CLASSES = np.arange(N_ACTION_CLASSES, dtype=SMALL_INT)


def solve_mask_and_map(choices, effect_index=0):
    """
    From a list of Consequences involving Pieces derive:
        (1) a mask over the array of Piece identities used by the
            selection methods; and,
        (2) a hash table to translate the selected Piece identity
            back into its Consequence.

    effect_index supports performing this conversion on specific
    Effects in multi-effect Consequences---e.g., Acquisitions
    which can entail gaining a Piece and entail trashing a Piece.
    """
    choice_mask = np.zeros(N_PIECE_IDENTITIES, dtype=BOOL)
    choice_map = dict()
    for choice in choices:
        total_order_index = choice.effects[effect_index].kwargs['piece'].total_order_index
        choice_mask[total_order_index] = True
        choice_map[total_order_index] = choice
    return choice_mask, choice_map


def solve_base_pmf_to_use(phenotype):
    """ Swap to endgame preferences based on the turn-ordinal. """
    if not(phenotype.use_endgame_pmf):
        return phenotype.pmf
    return phenotype.end_pmf

@njit
def solve_induced_pmf(mask, source_pmf):
    """
    Redistribute the weight in a full pmf across a subset of available
    options.
    """
    result = source_pmf
    result = result[mask]
    result /= result.sum()
    return result

@njit
def select_action_class(pmf_to_use, option_mask):
    return np.random.choice(ACTION_CLASSES[option_mask],
                            None,
                            solve_induced_pmf(option_mask, pmf_to_use))

@njit
def select_option(pmf_to_use, option_mask):
    return np.random.choice(PIECE_IDENTITIES[option_mask],
                            None,
                            solve_induced_pmf(option_mask, pmf_to_use))

def select_purchase_preferences(phenotype):
    pmf_to_use = solve_base_pmf_to_use(phenotype)
    return np.random.choice(PIECE_IDENTITIES,
                            size=N_PURCHASE_PREFERENCES,
                            replace=False,
                            p=pmf_to_use)

def select_option_inverse(pmf, option_mask):
    """
    To simulate having an inverted set of preferences, choose an
    available option according to preferences and remove it from
    consideration, until none remain. The final option chosen is
    considered the least preferred.
    """
    n_options_available = PIECE_IDENTITIES[option_mask].size
    pmf_to_use = solve_induced_pmf(option_mask, pmf)
    rankings = np.random.choice(PIECE_IDENTITIES[option_mask], size=n_options_available, replace=False, p=pmf_to_use)
    return rankings[-1]



class RandomStrategy(Agent):
    """
    Useful for testing features outside of strategic selection,
    or for comparing evolved strategies against random play.
    Other than automatically choosing forced options and playing
    all available Treasures, every choice is made at random with
    uniform probability over the available options.
    """
    def select(self, choices, decision):
        n_choices = len(choices)
        automatic_choice = self._automatic(choices, decision)
        if (automatic_choice is not None):
            return automatic_choice
        return choices[np.random.randint(n_choices)]


class EchoRandomStrategy(Echo, RandomStrategy):
    def __repr__(self):
        return "RNGStrategy {}".format(self.pid)


class Strategy(Agent):
    """
    Vehicle for playing Dominion given the strategy encoded in an
    evolved Phenotype.

    Heuristics:
        Always reveals Moat when attacked by Militia.

        Cellar only produces the option of discarding all of the
        Victory cards in hand.

        Limited form of look-ahead overrides evolved preferences
        when choosing between Acquisitions. Will ignore preferences
        if there is a choice that represents a forced win or a forced
        tie; will avoid choosing options which are forced losses.
        See: meta_select

        So long as the game isn't one action away from ending,
        selects actions to play based on the action's category rather
        than its identity. Otherwise, will look ahead at the
        consequences of each available action and automatically select
        forced wins or forced ties while avoiding forced losses.
        See: meta_meta_select

        Exhibits willingness to pass during the Buy Phase by sampling
        cards according to evolved preferences and then passing unless
        one of the sampled cards is an available option.
        See: select_purchase_acquisition
    """
    __slots__ = "phenotype",
    def __init__(self, pid, phenotype):
        super().__init__(pid=pid)
        self.phenotype = phenotype

    def refresh(self):
        super().refresh()
        self.phenotype.use_endgame_pmf = False

    def increment_turn_count(self):
        super().increment_turn_count()
        # Support turn-ordinal-based swapping to endgame preferences.
        if (self.n_turns_played > self.phenotype.genome.switch_index):
            self.phenotype.use_endgame_pmf = True

    def select_action(self, action_choices):
        action_class_option_mask = np.zeros(N_ACTION_CLASSES, dtype=BOOL)
        action_class_to_action_choice_map = {i:list() for i in range(N_ACTION_CLASSES)}

        for action_choice in action_choices:
            action_choice_action_class = action_choice.effects[0].kwargs['piece'].action_class
            action_class_to_action_choice_map[action_choice_action_class].append(action_choice)
            action_class_option_mask[action_choice_action_class] = True
        chosen_action_class = select_action_class(self.phenotype.action_class_pmf, action_class_option_mask)
        actions_in_chosen_action_class = action_class_to_action_choice_map[chosen_action_class]
        np.random.shuffle(actions_in_chosen_action_class)
        return actions_in_chosen_action_class[0]

    def select_discard(self, discard_choices):
        discard_option_mask, discardable_map = solve_mask_and_map(discard_choices)
        chosen_discardable_index = select_option_inverse(self.phenotype.pmf, discard_option_mask)
        return discardable_map[chosen_discardable_index]

    def select_independent_acquisition(self, acquisitions, pass_choice=None):
        """
        Choose from among Acquisitions which don't require Trashing.
        """
        if (len(acquisitions) == 1):
            return acquisitions[0]
        gainable_option_mask, gainable_map = solve_mask_and_map(acquisitions)
        chosen_index = select_option(solve_base_pmf_to_use(self.phenotype), gainable_option_mask)
        return gainable_map[chosen_index]

    def select_purchase_acquisition(self, acquisitions, pass_choice):
        """
        Choose from among Acquisitions by taking a sample without
        replacement from the current preference pmf. When none of
        the acquisitions entail gaining any of the sampled cards,
        pass instead of buying at random.

        Buying 0 cost cards simply because they are available is
        thereby avoided unless a legitimate preference for such
        cards is selected for---and, selection pressure is exerted
        such that endgame strategies should prefer victory point
        cards.
        """
        gainable_map = dict()
        for acquisition in acquisitions:
            gainable_toi = acquisition.gained_piece.total_order_index
            gainable_map[gainable_toi] = acquisition
        purchase_preferences = select_purchase_preferences(self.phenotype)
        for preferred_total_order_index in purchase_preferences:
            if (preferred_total_order_index in gainable_map):
                return gainable_map[preferred_total_order_index]
        return pass_choice

    def select_dependent_acquisition(self, acquisitions, pass_choice):
        """
        Choose from among Acquisitions which require Trashing, along
        with a potential option to do nothing instead.
        """
        if (len(acquisitions) == 1):
            return acquisitions[0]

        acquisition_map = dict()
        gainable_option_mask = np.zeros(N_PIECE_IDENTITIES, dtype=BOOL)
        trashable_option_mask = np.zeros(N_PIECE_IDENTITIES, dtype=BOOL)
        gainable_toi_to_trashable_toi_map = dict()
        for acquisition in acquisitions:
            gainable_toi = acquisition.gained_piece.total_order_index
            trashable_toi = acquisition.lost_piece.total_order_index
            acquisition_map[(gainable_toi, trashable_toi)] = acquisition
            if not(gainable_toi in gainable_toi_to_trashable_toi_map):
                gainable_option_mask[gainable_toi] = True
                gainable_toi_to_trashable_toi_map[gainable_toi] = []
            gainable_toi_to_trashable_toi_map[gainable_toi].append(trashable_toi)
        chosen_gainable_toi = select_option(solve_base_pmf_to_use(self.phenotype), gainable_option_mask)

        trashable_toi_candidates = gainable_toi_to_trashable_toi_map[chosen_gainable_toi]
        for trashable_toi_candidate in trashable_toi_candidates:
            trashable_option_mask[trashable_toi_candidate] = True
        pmf_to_use = self.phenotype.pmf if not(self.phenotype.use_endgame_pmf) else self.phenotype.end_pmf
        chosen_trashable_toi = select_option_inverse(pmf_to_use, trashable_option_mask)
        return acquisition_map[(chosen_gainable_toi, chosen_trashable_toi)]


    def meta_select(self, acquisitions, pass_choice, selection_method):
        """
        Override preference based selection with a naive look-ahead
        heuristic which prioritizes selecting Acquisitions which
        will end the game this turn while leaving the Strategy in a
        position where they are either winning outright (win) or tied
        for first place (tie). If no such Acquisitions are available,
        then selection proceeds according to evolved preferences, with
        the caveat that Acquisitions which will end the game this turn
        turn while leaving the Strategy in a position where they are
        not in first place (loss) are avoided if possible. A neutral
        Acquisition is one that will not trigger the end of the game
        this turn.
        """
        if not(self.state.almost_over):
            return selection_method(acquisitions, pass_choice)

        wins, ties, losses, neutrals = evaluate_acquisitions(self, acquisitions)

        # Case: Take an outright win if possible.
        if wins:
            return wins[0]

        # Case: Take a tie for first if possible.
        elif ties:
            return ties[0]

        # Case: Only losses are available. Lose in an arbitrary
        #       manner unless passing is possible.
        elif not(neutrals):
            return pass_choice if (pass_choice is not None) else losses[0]

        # Case: Call the appropriate default selection method to choose
        #       between the neutral Acquisitions.
        return selection_method(neutrals, pass_choice)


    def meta_meta_select(self, acts, pass_choice):
        """
        Override or inform the process of selecting an action based on
        preferences with an analysis regarding wins, ties, and losses.

        If any of the actions will generate at least one consequence
        which forces a win, select it. Otherwise, if any of the actions
        generate at least one consequence forcing a tie for first,
        select it. When no win or tie is available, before subjecting
        the actions to selection via preference, first filter out any
        choices which exclusively generate consequences which are
        forced losses. In the event that this process of removing
        forced losses leaves nothing to select between, return the
        choice of passing the action phase.
        """
        if not(self.state.almost_over):
            return self.select_action(acts)

        acts_to_consider = []
        acts_tying_for_first = []
        might_gain, cant_gain = classify_acts(acts)
        acts_to_consider.extend(cant_gain)
        for act in might_gain:
            choices = expand_choices(state=self.state, actor=self, decision=act.action.decision)
            passes, nulls, acquisitions, other = extract_acquisitions(choices)
            # Case: Playing act will generate one or more Acquisitions.
            if acquisitions:
                wins, ties, losses, neutrals = evaluate_acquisitions(self, acquisitions)
                # Short-circuit if an act which generates an Acquisition which will end the game
                # this turn with this Strategy winning outright is found.
                if wins:
                    return act
                # If an act which generates an Acquisition which will end the game this turn with
                # this Strategy tied for first place is found, store it but wait to see if any
                # outright wins exist.
                elif ties:
                    acts_tying_for_first.append(act)
                # If there aren't any neutrals, playing this act will exclusively generate
                # Acquisitions which will end the game this turn with this Strategy below
                # first place; otherwise, leave it available for default selection.
                elif neutrals:
                    acts_to_consider.append(act)

            # Playing this act won't generate Acquisitions so can't cause a loss.
            else:
                acts_to_consider.append(act)

        # Didn't short-circuit on a forced outright win. Any ties for first?
        if acts_tying_for_first:
            return acts_tying_for_first[0]

        # At this point, no acts_to_consider means that each act was a potential generator of
        # Acquisitions which exclusively generated Acquisitions which will end the game this
        # turn with this Strategy below first place. Select the pass_action_phase_option to
        # avoid losing.
        elif not(acts_to_consider):
            return pass_choice

        # Subject the acts_to_consider to default selection.
        return self.select_action(acts_to_consider)


    def select(self, choices, decision):
        automatic_choice = self._automatic(choices, decision)
        if (automatic_choice is not None):
            return automatic_choice

        # Case: Which action to play during the action phase.
        if isinstance(decision, ActionChoices):
            return self.meta_meta_select(acts=choices[1:], pass_choice=choices[0])

        # Case: Always discard the maximum number of victory cards with Cellar.
        elif isinstance(decision, CellarChoices):
            # NOTE #
            # choices[0] will discard zero cards and draw zero cards.
            # choices[1] will discard all of the victory cards available and draw that many cards.
            # choices[1] is guaranteed to exist by the single choice short-circuit above.
            return choices[1]

        # Case: Discard cards to an opponent's Militia according to inverse early game pmf.
        elif isinstance(decision, MilitiaVictimProcedure):
            return self.select_discard(choices)

        # Case: The only other types of Decisions are those which can potentially generate
        #       Acquisitions which are the only Consequences which can end up triggering the
        #       end of the game.
        passes, nulls, purchases, dependents, acquisitions, other = classify_consequences(choices)

        if passes:
            pass_choice = passes[0]
        elif nulls:
            pass_choice = nulls[0]
        else:
            pass_choice = None

        if purchases:
            acquisitions = purchases
            selection_method = self.select_purchase_acquisition
        elif dependents:
            acquisitions = dependents
            selection_method = self.select_dependent_acquisition
        elif acquisitions:
            acquisitions = acquisitions
            selection_method = self.select_independent_acquisition
        return self.meta_select(acquisitions, pass_choice, selection_method)


    def record_pmf_data(self):
        """
        Produce a minimal representation of the parameters which govern the
        Strategy's behavior.
        """
        return [[self.phenotype.genome.alpha, self.phenotype.genome.sigma],
                [self.phenotype.genome.end_alpha, self.phenotype.genome.end_sigma],
                [self.phenotype.genome.action_class_alpha, self.phenotype.genome.action_class_sigma],
                self.phenotype.genome.switch_index]


class EchoStrategy(Echo, Strategy):
    def __repr__(self):
        return "Strategy {}".format(self.pid)
