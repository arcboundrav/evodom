import numpy as np

from numba import njit
from time import perf_counter
from multiprocessing import Pool

from evolvedominion.utils import CACHE_MANAGER
from evolvedominion.agents.strategy import Strategy
from evolvedominion.algorithm.tournaments import Seeding, Elimination
from evolvedominion.params import (
    EPSILON,
    N_PROCESSES,
    SWITCH_INDEX,
    GROUPSIZE,
    N_PIECE_IDENTITIES,
    N_ACTION_CLASSES,
)
from evolvedominion.types import (
    FLOAT,
    INT,
    SMALL_INT,
    BOOL,
)
from evolvedominion.display.text import (
    display_generation_duration,
    display_simulation_duration,
)


# Penalize preferences which put weight on victory cards before the end game
# or put weight on acquiring Curse at any point.
MAX_PENALTY = 1000000

# The top % of phenotypes (based on fitness evaluation) which will participate
# in the creation of the next generation.
PERCENT_ANCESTORS = 0.5

# Percent of the next generation derived by cloning the fittest ancestors.
PERCENT_VIA_CLONING = 0.25

# Percent of the next generation derived by recombining the ancestors.
PERCENT_VIA_RECOMBINATION = 0.5

# Percent of the next generation derived by spawning novel strategies.
PERCENT_VIA_NOVELTY = 1.0 - (PERCENT_VIA_CLONING + PERCENT_VIA_RECOMBINATION)

# Probability offspring inherits an alpha parameter from the fitter parent.
P_INHERIT_ALPHA_TOP = 0.85

# Probability offspring inherits a sigma parameter from the fitter parent.
P_INHERIT_SIGMA_TOP = 0.85

# Probability offpsring inherits a sigma parameter from the less fit parent.
P_INHERIT_SIGMA_BOT = 1.00 - P_INHERIT_SIGMA_TOP

# Probability that an alpha parameter mutates during cloning or reproduction.
P_ALPHA_CHANGE = 0.1

# Probability that a sigma parameter mutates during cloning or reproduction.
P_SIGMA_CHANGE = 0.1
P_ACTION_CLASS_SIGMA_CHANGE = 0.01

# Cap on the magnitude of changes to alpha parameters via mutation.
MAX_ALPHA_JUMP = 5

# Experimentation and profiling agree:
# An action class pmf which results in nearly deterministic sequencing of actions as
# a function of their action class is sufficient for evolving useful strategies.
STRICT_ACTION_CLASS_PREFERENCE_PMF = np.array([0.999-EPSILON, 0.001, EPSILON], dtype=FLOAT)
ACTION_CLASS_MODELSPACE = {0:STRICT_ACTION_CLASS_PREFERENCE_PMF}

# Load the set of preference pmfs whose product with the factorial(N_PIECE_IDENTITIES)
# possible permutations of the sigma parameter constitutes the preference pmf search space
# for the algorithm.
MODELSPACE = CACHE_MANAGER.load_cache()

# Number of distinct base pmfs indexed by alpha parameter.
N_ALPHA = len(MODELSPACE)

# The closed interval of indices within which alpha parameters are free to vary.
MODELSPACE_INDICES = list(range(N_ALPHA))

# Defines the upper bound on alpha values.
MAX_ALPHA = N_ALPHA - 1

# Defines a reference parameter from which all sigma parameters are derived.
INITIAL_SIGMA = np.arange(N_PIECE_IDENTITIES, dtype=SMALL_INT)

# Defines a reference parameter from which all action class sigma parameters are derived.
INITIAL_ACTION_CLASS_SIGMA = np.arange(N_ACTION_CLASSES, dtype=SMALL_INT)


class Phenotype:
    """
    Maps genetic parameters to behavioral dispositions and informs
    Strategy choices.
    """
    __slots__ = (
        "genome",
        "use_endgame_pmf",
        "pmf",
        "end_pmf",
        "action_class_pmf",
    )
    def __init__(self, genome):
        self.genome = genome
        self.use_endgame_pmf = False
        self.pmf = MODELSPACE[genome.alpha][genome.sigma]
        self.end_pmf = MODELSPACE[genome.end_alpha][genome.end_sigma]
        self.action_class_pmf = ACTION_CLASS_MODELSPACE[genome.action_class_alpha][genome.action_class_sigma]


class Genome:
    """
    Encodes parameters which ultimately govern Strategy behavior.
    Supports recombination and cloning to derive new instances.
    Assigned a rank quantifying fitness relative to a given generation.
    """
    __slots__ = (
        "alpha",
        "sigma",
        "end_alpha",
        "end_sigma",
        "action_class_alpha",
        "action_class_sigma",
        "rank",
        "switch_index",
    )
    def __init__(self, alpha, sigma, end_alpha, end_sigma, action_class_alpha, action_class_sigma):
        self.alpha = alpha
        self.sigma = sigma
        self.end_alpha = end_alpha
        self.end_sigma = end_sigma
        self.action_class_alpha = action_class_alpha
        self.action_class_sigma = action_class_sigma
        self.rank = 0
        self.switch_index = SWITCH_INDEX


def random_initial_sigma():
    np.random.shuffle(INITIAL_SIGMA)
    return np.array(INITIAL_SIGMA)


def random_initial_action_class_sigma():
    np.random.shuffle(INITIAL_ACTION_CLASS_SIGMA)
    return np.array(INITIAL_ACTION_CLASS_SIGMA)


@njit
def roll(p):
    return (np.random.random() <= p)


@njit
def delta_direction():
    return 1 if (np.random.random() <= 0.5) else -1


@njit
def mutate_preference_alpha(alpha):
    if roll(P_ALPHA_CHANGE):
        delta = delta_direction() * np.random.randint(1, MAX_ALPHA_JUMP)
        new_alpha = alpha + delta
        new_alpha = max(0, min(new_alpha, MAX_ALPHA))
        return new_alpha
    return alpha


def mutate_action_class_alpha(alpha):
    return alpha


@njit
def mutate_sigma(sigma, indices):
    if roll(P_SIGMA_CHANGE):
        i, j = np.random.choice(indices, 2, False)
        tmp = sigma[i]
        sigma[i] = sigma[j]
        sigma[j] = tmp


@njit
def mutate_action_class_sigma(sigma):
    if roll(P_ACTION_CLASS_SIGMA_CHANGE):
        i, j = np.random.choice(INITIAL_ACTION_CLASS_SIGMA, 2, False)
        tmp = sigma[i]
        sigma[i] = sigma[j]
        sigma[j] = tmp


@njit
def recombine_sigma(top, bot, N=17):
    """
    Combine two permutations of size N into one permutation, R.
    For each index, i, into R:
        1) Select which parent will contribute the element at R[i].
        2) Find the leftmost element of the contributing parent
           which has not yet been included in R.
           If such an element exists, insert it into R[i].
           Otherwise, finish filling result with the entries of
           the other parent which have not yet been included in R.
    """
    top_idx = 0
    bot_idx = 0
    i = 0
    j = 0
    sigma_data = np.zeros(N, dtype=SMALL_INT)
    new_sigma = np.empty(N, dtype=SMALL_INT)
    for i in range(N):
        if roll(P_INHERIT_SIGMA_BOT):
            while ((bot_idx < N) and (sigma_data[bot[bot_idx]])):
                bot_idx = bot_idx + 1
            if (bot_idx < N):
                this_value = bot[bot_idx]
                sigma_data[this_value] = 1
                new_sigma[i] = this_value
                bot_idx = bot_idx + 1
            else:
                while (top_idx < N):
                    new_sigma[i] = top[top_idx]
                    i = i + 1
                    top_idx = top_idx + 1
        else:
            while ((top_idx < N) and (sigma_data[top[top_idx]])):
                top_idx = top_idx + 1
            if (top_idx < N):
                this_value = top[top_idx]
                sigma_data[this_value] = 1
                new_sigma[i] = this_value
                top_idx = top_idx + 1
            else:
                while (bot_idx < N):
                    new_sigma[i] = bot[bot_idx]
                    i = i + 1
                    bot_idx = bot_idx + 1
    return new_sigma


def recombine(a, b):
    """
    Derive an offspring given two ancestor Genomes. Offspring
    are more likely to inherit attributes from the fitter of
    the two parents.
    """
    top, bot = (a, b) if (a.rank <= b.rank) else (b, a)
    new_alpha = top.alpha if roll(P_INHERIT_ALPHA_TOP) else bot.alpha
    new_end_alpha = top.end_alpha if roll(P_INHERIT_ALPHA_TOP) else bot.end_alpha
    new_action_class_alpha = top.action_class_alpha

    new_alpha = mutate_preference_alpha(new_alpha)
    new_end_alpha = mutate_preference_alpha(new_alpha)

    new_sigma = recombine_sigma(top.sigma, bot.sigma)
    new_end_sigma = recombine_sigma(top.end_sigma, bot.end_sigma)
    new_action_class_sigma = top.action_class_sigma

    mutate_sigma(new_sigma, INITIAL_SIGMA)
    mutate_sigma(new_end_sigma, INITIAL_SIGMA)
    mutate_action_class_sigma(new_action_class_sigma)

    result = Genome(alpha=new_alpha, sigma=new_sigma,
                    end_alpha=new_end_alpha, end_sigma=new_end_sigma,
                    action_class_alpha=new_action_class_alpha, action_class_sigma=new_action_class_sigma)
    return result



def clone_genome(a):
    new_alpha = mutate_preference_alpha(a.alpha)
    new_sigma = np.array(a.sigma)
    new_end_alpha = mutate_preference_alpha(a.alpha)
    new_end_sigma = np.array(a.end_sigma)
    new_action_class_alpha = a.action_class_alpha
    new_action_class_sigma = np.array(a.action_class_sigma)
    mutate_sigma(new_sigma, INITIAL_SIGMA)
    mutate_sigma(new_end_sigma, INITIAL_SIGMA)
    mutate_action_class_sigma(new_action_class_sigma)
    result = Genome(alpha=new_alpha, sigma=new_sigma,
                    end_alpha=new_end_alpha, end_sigma=new_end_sigma,
                    action_class_alpha=new_action_class_alpha, action_class_sigma=new_action_class_sigma)
    return result


@njit
def recombination_schedule(n_offspring, n_ancestors):
    """
    Sample n_offspring pairs of distinct ancestor indices with uniform
    probability.

    Making a single call to np.random.random_sample((n_row, n_col))
    then transforming the sampled values in each row into unique
    indices via argsort is equivalent to generating n_row permutations
    of the integers in the closed interval [0, ncol-1].

    Flattening the result and iterating over the entries 2 at a time is
    roughly equivalent to exhaustively sampling pairs of ancestors
    without replacement, restarting the procedure only after every
    ancestor has had a chance to find a pair, until n_offspring pairs
    have been generated.

    Profiling demonstrates that this approach is much faster than
    repeatedly calling np.random.choice() with replace=False and
    size=2.

    Note: argsort doesn't currently play nicely with numba, so that
    step occurs in the Simulation method which calls this auxillary
    function.
    """
    pairs_per_row = int(n_ancestors // 2)
    n_rows = max(1, np.ceil(n_offspring / pairs_per_row))
    result = np.random.random_sample((n_rows, n_ancestors))
    return result


@njit
def penalize(pmf, end_pmf):
    return MAX_PENALTY * (np.sum(pmf[0:3]) + end_pmf[0])


class Simulation:
    __slots__ = (
        "simname",
        "N",
        "k",
        "N_ANCESTORS",
        "N_CLONE",
        "N_RECOMBINE",
        "N_NOVEL",
        "alphas",
        "genomes",
        "winning_phenotypes",
        "players",
        "seeding_tournament",
        "elimination_tournament",
        "phenotype_record",
        "current_generation_index",
        "end_alphas",
        "action_class_alphas",
    )
    def __init__(self, simname, N):
        self.simname = simname
        self.N = N
        self.k = GROUPSIZE
        self.N_ANCESTORS = int(self.N * PERCENT_ANCESTORS)
        self.N_CLONE = self._solve_N_CLONE()
        self.N_RECOMBINE = self._solve_N_RECOMBINE()
        self.N_NOVEL = self._solve_N_NOVEL()
        self.alphas = np.array(MODELSPACE_INDICES, dtype=INT)
        self.end_alphas = np.array(MODELSPACE_INDICES, dtype=INT)
        self.genomes = list()
        self.winning_phenotypes = list()
        self.players = [Strategy(pid=i, phenotype=None) for i in range(N)]
        self.seeding_tournament = Seeding(N)
        self.elimination_tournament = Elimination(N)
        self.phenotype_record = dict()
        self.current_generation_index = 0

    def _solve_N_CLONE(self):
        if self.N_ANCESTORS:
            return int(self.N * PERCENT_VIA_CLONING)
        return 0

    def _solve_N_RECOMBINE(self):
        if self.N_ANCESTORS:
            return int(self.N * PERCENT_VIA_RECOMBINATION)
        return 0

    def _solve_N_NOVEL(self):
        return self.N - (self.N_CLONE + self.N_RECOMBINE)

    def record_winning_phenotype(self):
        self.winning_phenotypes.append(self.elimination_tournament.entrants[0].phenotype)

    def record_phenotype_pmfs(self):
        self.phenotype_record[self.current_generation_index] = [player.record_pmf_data() for player in self.players]

    def data(self):
        return {
            'genomes':self.genomes,
            'winners':self.winning_phenotypes,
            'record':self.phenotype_record,
        }

    def random_spawn(self, N=None):
        """ Generate N Genomes with randomized attributes. """
        N = self.N if (N is None) else N
        np.random.shuffle(self.alphas)
        np.random.shuffle(self.end_alphas)
        for i in range(N):
            j = i % N_ALPHA
            self.genomes.append(Genome(alpha=self.alphas[j],
                                       sigma=random_initial_sigma(),
                                       end_alpha=self.end_alphas[j],
                                       end_sigma=random_initial_sigma(),
                                       action_class_alpha=0,
                                       action_class_sigma=random_initial_action_class_sigma()))

    def update_players(self):
        """
        Re-use the same Strategy instances by replacing their
        Phenotypes with the latest generation of Genomes.
        """
        for i in range(self.N):
            self.players[i].phenotype = Phenotype(self.genomes[i])

    def _sort_entrants_by_fitness_and_penalty(self, entrants):
        """
        Passes the pmf and end_pmf attributes of each entrant's
        Phenotype to penalize() in order to play nicely with numba.
        """
        entrants_and_penalties = []
        PENALTY_INDEX = 1
        for entrant in entrants:
            penalty = penalize(entrant.phenotype.pmf, entrant.phenotype.end_pmf)
            entrants_and_penalties.append((entrant, penalty))
        entrants_and_penalties.sort(key=lambda entrant_and_penalty: (entrant_and_penalty[PENALTY_INDEX]))
        return entrants_and_penalties

    def _rerank_genomes(self, entrants):
        """
        Expects entrants to be sorted ascending by rank, where a lower
        rank implies higher fitness. Taking fitness and penalties into
        account, re-rank entrants in a stable manner, i.e., given two
        entrants A and B:
        If fitness(A)>fitness(B) and penalty(A)==penalty(B), then
        index(A)<index(B) both before and after the re-ranking.

        Overwrites the rank of each Entrant's genome to reflect the
        new order and returns them.
        """
        penalized_genomes = []
        ENTRANT_INDEX = 0
        entrants_and_penalties = self._sort_entrants_by_fitness_and_penalty(entrants)
        for i, entrant_and_penalty in enumerate(entrants_and_penalties):
            penalized_genome = entrant_and_penalty[ENTRANT_INDEX].phenotype.genome
            penalized_genome.rank = i
            penalized_genomes.append(penalized_genome)
        return penalized_genomes

    def solve_ancestors(self):
        """
        Cull unfit Genomes to leave the subset which will have a chance
        to pass on their attributes to the next generation via cloning
        and reproduction.
        """
        if self.N_ANCESTORS:
            reranked_genomes = self._rerank_genomes(self.elimination_tournament.entrants)
            return reranked_genomes[:self.N_ANCESTORS]
        return []

    def clone_ancestors(self, ancestors):
        """
        Pass on the the top self.N_CLONE Genomes to the next generation,
        subject to mutation.
        """
        if self.N_CLONE:
            self.genomes.extend(clone_genome(ancestors[i]) for i in range(self.N_CLONE))

    def _recombine_given_schedule(self, ancestors, schedule):
        """
        A schedule is a sequence of index pairs specifying which ancestors
        will reproduce to add an offspring to the next generation.
        """
        indices = ((schedule[j], schedule[j+1]) for j in range(0, schedule.size, 2))
        for i in range(self.N_RECOMBINE):
            a, b = next(indices)
            self.genomes.append(recombine(ancestors[a], ancestors[b]))

    def recombine_ancestors(self, ancestors):
        """ See: recombination_schedule """
        if self.N_RECOMBINE:
            if (self.N_ANCESTORS == 1):
                schedule = np.zeros((2 * self.N_RECOMBINE))
            else:
                schedule = np.argsort(recombination_schedule(self.N_RECOMBINE, self.N_ANCESTORS)).ravel()
            self._recombine_given_schedule(ancestors, schedule)

    def inject_novelty(self):
        """
        Discourage the entire population converging to a local maxima
        by injecting entirely random Genomes into each generation.
        """
        self.random_spawn(N=self.N_NOVEL)

    def solve_next_generation(self):
        self.genomes.clear()
        ancestors = self.solve_ancestors()
        if ancestors:
            self.clone_ancestors(ancestors)
            self.recombine_ancestors(ancestors)
        self.inject_novelty()

    def evaluate_fitness(self, pool):
        """
        fitness(Strategy A) > fitness(Strategy B) when Strategy A
        places higher than Strategy B in a single elimination
        tournament wherein competitors play Dominion.
        """
        self.seeding_tournament.run(entrants=self.players, pool=pool)
        self.elimination_tournament.run(entrants=self.seeding_tournament.entrants, pool=pool)

    def generate(self, pool):
        self.update_players()
        self.evaluate_fitness(pool)
        self.record_winning_phenotype()
        self.record_phenotype_pmfs()
        self.solve_next_generation()

    def evolve(self, n_generation, debug=False):
        self.random_spawn()
        if debug:
            with Pool(N_PROCESSES) as pool:
                self.generate(pool)
        else:
            t0 = perf_counter()
            with Pool(N_PROCESSES) as pool:
                for i in range(n_generation):
                    t0_ = perf_counter()
                    self.generate(pool)
                    display_generation_duration(generation_index=i, n_generation=n_generation, t1=perf_counter(), t0=t0_)
                    self.current_generation_index = self.current_generation_index + 1
            display_simulation_duration(simname=self.simname, t1=perf_counter(), t0=t0)
