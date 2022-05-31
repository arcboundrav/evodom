#################
# I/O Constants #
#################
APPAUTHOR = "StatisticalStrategies"

APPNAME = "evolvedominion"

MODELSPACE_CACHE_FILENAME = "fullmodelspace.pkl"


#########################
# Simulation Parameters #
#########################
# Minimum amount of weight assigned to an entry in a pmf.
EPSILON = 0.000000001

# Turn ordinal when strategies swap to an endgame acquisition strategy.
SWITCH_INDEX = 12

# Number of distinct Piece types a Strategy will sample to decide between
# purchasing a Piece or passing the Buy Phase. A higher number encodes less
# willingness to pass.
N_PURCHASE_PREFERENCES = 3

# Number of Piece types
N_PIECE_IDENTITIES = 17

# Number of Action Classes
N_ACTION_CLASSES = 3

# Number of worker processes to use during simulations.
N_PROCESSES = 8

# Number of players per game.
GROUPSIZE = 4

# Minimum number of genomes to spawn per generation.
MIN_NSTRAT = 8

# Maximum number of genomes to spawn per generation.
MAX_NSTRAT = 512

# Default number of genomes to spawn per generation.
DEFAULT_NSTRAT = 128

# Minimum number of generations to evolve per simulation.
MIN_NGEN = 1

# Maximum number of generations to evolve per simulation.
MAX_NGEN = 9999

# Default number of generations to evolve per simulation.
DEFAULT_NGEN = 20


#########
# Hints #
#########
# Failure message for attempting to run a simulation with a previously used name without
# passing -o to enable overwriting data.
INVALID_OVERWRITE = "Run with option -o to re-use simulation name: {} (and overwrite prior data)."

# CLI option description: -o
DEFAULT_OVERWRITE_HELP = "Overwrite past data associated with simname."

# Invalid argument warning for simulation name.
INVALID_SIMNAME = "Simulation names must consist exclusively of alphanumeric characters."

# CLI option description: DEFAULT_NSTRAT
DEFAULT_NSTRAT_HELP = "Number of strategies evolved each generation. (Default:{})"
DEFAULT_NSTRAT_HELP = DEFAULT_NSTRAT_HELP.format(DEFAULT_NSTRAT)

# Invalid argument warning for number of genomes.
INVALID_NSTRAT = "nstrat must be in the closed interval [{},{}] and evenly divisible by {}."
INVALID_NSTRAT = INVALID_NSTRAT.format(MIN_NSTRAT, MAX_NSTRAT, GROUPSIZE)

# CLI option description: DEFAULT_NGEN
DEFAULT_NGEN_HELP = "Number of generations evolved by the genetic algorithm. (Default:{})"
DEFAULT_NGEN_HELP = DEFAULT_NGEN_HELP.format(DEFAULT_NGEN)

# Invalid argument warning for number of generations.
INVALID_NGEN = "ngen must be an integer in the closed interval [{},{}]."
INVALID_NGEN = INVALID_NGEN.format(MIN_NGEN, MAX_NGEN)


#########
# Flags #
#########
# An action generates at least one consequence that is a forced win.
WIN = 0

# An action generates at least one consequence that is a forced tie, but no forced wins.
TIE = 1

# An action generates at least one consequence that is not a forced loss; no forced wins or forced ties.
INCLUDE = 2

# An action exclusively generates forced losses.
EXCLUDE = 3

# An action with a consequence which increases the number of actions to take.
NONTERMINAL = 0

# An action which does not increase the number of actions to take, but can draw 1 or more cards.
TERMINAL_WITH_DRAW = 1

# An action which does not increase the number of actions nor draws cards.
TERMINAL_WITHOUT_DRAW = 2

ACTION_PHASE = "Action Phase"

TREASURE_PHASE = "Treasure Phase"

BUY_PHASE = "Buy Phase"

PHASES = set([
    ACTION_PHASE,
    TREASURE_PHASE,
    BUY_PHASE,
])
