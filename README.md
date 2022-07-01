# evolvedominion
A text-based interface for evolving—and, playing against—strategies for Dominion.

This project was created as a proof of concept that minimally sophisticated agents
which rely only on local information could attain competent play through the use
of a genetic algorithm.

![Tests](https://github.com/arcboundrav/evodom/actions/workflows/tests.yml/badge.svg)

## Installation

Before installing it is recommended to create and activate a [virtualenv](https://docs.python.org/3/tutorial/venv.html) using a version of [Python](https://www.python.org/downloads/) >= 3.8.12.

### Install from PyPI using `pip`
```
python -m pip install -U pip
python -m pip install -U evolvedominion
```

### Install from Git
Clone the repository using either
```
git clone https://github.com/evolvedominion/evolvedominion.git
```
or
```
git clone git@github.com:evolvedominion/evolvedominion.git
```
Navigate to the top level of the package and install using `pip`
```
cd evolvedominion
python -m pip install .
```
Note: The tests will require additional dependencies.
```
python -m pip install -r requirements_dev.txt
```
Updating evolvedominion to the latest release can be done
by navigating to the repository and using `git pull`

## Example

After installation evolvedominion can be run from the command line.

### Evolving Strategies

Strategies need to be evolved prior to being able to play against them.
The following command will run the genetic algorithm for `100` generations
with a population size of `128`. The results will be stored using the key
`demo`. This key will be used later to play against the best ones.
```
evodom evolve demo --ngen 100 --nstrat 128
```
Note: The number of Strategies per generation must be in the closed
interval `[8, 512]` and be evenly divisible by `4`. The algorithm is capped at `9999`
generations per run. Experience shows that a few hundred generations is more
than adequate to evolve competent Strategies.

Note: Passing the `-o` flag permits the overwriting of past results to reuse
a key. Valid keys are non-empty alphanumeric strings. Data is saved in a
platform specific location—e.g., `~/.local/share/evolvedominion/` on Ubuntu.
Entering the following in the Python interpreter will print the path on your
machine:
    >>> import platformdirs
    >>> platformdirs.PlatformDirs().user_data_dir


### Playing Against Evolved Strategies

The following command will launch a text-based game against the three
strongest Strategies evolved under the key `demo`:
```
evodom play demo
```

### Interface

Type `?` at the prompt to list the available commands.
Exit the game early at any time with `CTRL+C`.


## Project Information

Key concepts and implementation details are elaborated upon below.


### A Minimalist Approach to Strategies

#### The Basics

Start with a set of mutually exclusive outcomes of size C called a sample space.
Any array A of size C satisfying the following constraints represents a discrete
probability distribution over the sample space:

    (Constraint 1) Each element A[i] falls between 0 and 1; and,
    (Constraint 2) The sum of the C elements of A is 1.

Label each outcome in the sample space by assigning it an integer in the closed
interval `[0, C-1]`. Then the outcome with label `i` will be sampled according to `A`
with probability `A[i]`.


#### Applying the Basics

There are `C = 17` types of Cards. We label each type by assigning
it an integer in the closed interval `[0, 16]`, then use that
interval as our sample space, which we call `I`. We store the mapping
from label to Card type implicitly in an array of Card types, `T`,
where `T[i]` is the Card type with label `i`.

We can represent our preferences for certain types of Cards with
a discrete probability distribution over `I`, which we call `P`. To
sample a Card type, we sample a label `i` in `I` according to `P`, then
access the corresponding Card type indexed by `i` in `T`.


#### Preferences

Consider the case where in the long run we'd like to exhibit no
bias toward any type of Card—i.e., we prefer each type of Card
equally.

Then `P` is a uniform distribution over `I` where every element of `P` is
equal to `1 / C`. In the limit of sampling `N >> C` Card types according
to `P`, the number of times each Card type is sampled will tend to the
same value, `N / C`.

A uniform probability distribution over `I` is conceptually equivalent
to a fair die with 17 faces where each face is labelled by a distinct
integer in `I`.

The default selection procedure is then analogous to rolling such a
die, noting the index `i` on the face that comes up, then returning the
Card type `T[i]`.

In contrast to a fair die, over many rolls the uneven distribution
of weight inside a loaded die causes certain outcomes to occur more
frequently than others. To represent preferences for certain Card
types relative to others it is therefore necessary for `P` to be a
non-uniform discrete probability distribution over `I`.

The space of discrete probability distributions searched by the genetic
algorithm is described [here](#model-spaces).



#### Normalization

Let `S` be a non-empty subset of `I` with size `M`. If `S` is a proper
subset of `I`—i.e., `1 <= M < |C|`—then the default selection procedure
must be adapted. From our base preferences `P`, we derive new preferences,
`P(S)`, which represent relative preferences for the elements of `I` which
occur in `S`. This procedure is called normalization.

Example

`S = [0, 1]`—i.e, only the first `M = 2` types of Cards in `T` are available
for selection.

Consider base preferences `P` where `P[0] == 0.05`, `P[1] == 0.15`, and `P[j]
(1 < j < C)` obey the constraints qualifying `P` as a probability distribution,
but are otherwise irrelevant.

`P(S)` will represent the base preferences adapted to the restricted set of
available options. To find `P(S)`:

Start with the subarray of `P` indexed by `S`

    P(S) := [P[0], P[1]] == [0.05, 0.15]

Compute its sum, `E`

    E    := sum(P(S))    == (0.05 + 0.15) == 0.2

Divide each element of `P(S)` by `E` to transform it into a valid discrete
probability distribution.

    P(S) := P(S) / E     == [0.25, 0.75]

To select a Card type from the restricted set of available options, follow
the default procedure—but replace `I` with `S` and replace `P` with `P(S)`.
That is, roll the re-weighted die to select from amongst only the available
options. Normalization means that `P(S)` will preserve the relative preferences
encoded in `P` between the available Card types.

In this example, during the default selection procedure when every option is
available, `T[1]` is three times more likely to be selected than `T[0]`. After
normalization, `T[1]` remains three times more likely to be selected than `T[0]`.

This procedure can fail unless the distributions obey a stronger version of
Constraint 1:

    0 < epsilon <= P[i] < (1 - ((N - 1) * epsilon))

In other words, the minimum weight for each Card type in any distribution
is epsilon—a small non-zero value—rather than zero.


#### Minimalist Strategies

Each Turn can be [decomposed into a series of selections](#decisions-consequences-and-effects). It follows that
equipping a Strategy with the ability to make selections is sufficient to
permit it to play a Game.

Usually a Strategy will make a selection using the procedure outlined
above. Each of the probability distributions which inform selection
represent a [preference](#preferences) for acquiring certain Card types.

In special cases, a Strategy will make a selection using a custom
procedure which still relies on sampling informed by evolved
preferences. These custom procedures—and, the naive heuristic
that a Strategy may deploy when the game is nearly over—are
discussed [here](#heuristics).


### Explorations

#### Switch Index

Early versions of the algorithm subjected the switch index parameter to natural selection.
It was permitted to vary from Turn 10 to Turn 30. Multiple runs of the algorithm converged
on dominant Strategies which had in common a switch index in the range 12-15. There was no
appreciable difference between Strategies evolved in regimes with the parameter fixed to
12 vs. 15, so 12 was settled on.


#### Seeding Tournament

Early versions of the algorithm used a Swiss tournament to seed the Elimination tournament
where the number of rounds in the Swiss tournament was given by log2(n_genomes). Profiling
indicated that the Swiss tournament scheduling algorithm was problematic. I discovered that
the dominant Strategies the genetic algorithm tended to converge to were not appreciably
different as the number of rounds in the Swiss tournament decreased, to the point where it
became clear that a single round was sufficient for seeding. More rigorous fitness evaluation
may be necessary in more sophisticated future extensions.


#### Model Spaces

##### Action Class

Early versions of this project included hand-tuned heuristics for
sequencing Action plays. To explore the viability of evolved
stochastic preferences, and in the spirit of minimalism, these were
removed and replaced with a predecessor of the current system, where
the search space for action class preference distributions was much
larger. Experimental results indicated that successful Strategies
almost exclusively evolved extremely biased distributions. Instead
of reverting to hard-coded sequencing heuristics, the current system
remains to support extensions to Action sequencing and as a proof of
concept of the viability of strict total order policies. Action
selection is discussed further [here](#action-class-preferences).


##### Preferences

Each preference distribution was originally a discretized member of the Dirichlet
distribution. Both the symmetric Dirichlet distribution and asymmetric Dirichlet
distribution were used.

Variability in Phenotypical expression was introduced by sampling a single member of the
distribution given the corresponding Genome's concentration parameters. This approach
yielded too much inter-generational variation. The utility of the fitness evaluation
of a Phenotype as a determinant of which Genomes should contribute to future generations
was nullified because on two successive generations the same Genome could be expressed
as Phenotypes exhibiting vastly different behavior.

Deriving the Phenotype's preference distribution by instead taking an expectation over
a larger sample from the Dirichlet distribution using a given Genome's concentration
parameters reduced this variability without removing it entirely.

Overall performance improved following a dramatic increase in the sparsity of the
parameter space. Visualization of the associated model space informed and
motivated this change. Beforehand, the redundancy was so extreme that traveling
large distances in the parameter space did not correlate strongly with meaningful
changes in behaviour.


#### Expressivity and Consistency

The minimalist approach of representing preferences using a single distribution comes
with a major tradeoff between consistency and expressivity.

By expressivity I mean the ability to encode an order of relative preferences
between 2 <= n <= N distinct options.

By consistency I mean the expected number of inversions in a sequence of N
samples without replacement from the distribution, relative to a descending
sort of the sample space's elements according to the magnitude of their
individual probabilities.

To wit: individual probabilities can be assigned so that non-trivial relative
preferences between every option are represented explicitly, but, the affine
constraint on the sum of the probabilities means that such a representation
will cause the distribution to approach uniformity as the number of total
options increases. The closer the distribution is to uniform, the less
consistent it becomes in terms of producing the expressed order of preferences.

This observation was one of the motivations for replacing the Dirichlet model
space with a much smaller set of distributions which encode relative
preferences between 3 <= n <= 5 options.

Yet, largely due to how well the game is balanced and how simple it is, the
minimalist approach works in practice. Specifically:

1) Which Cards are available for Acquisition usually varies in relation
   to how many Coins an Agent has.

2) An Agent is more likely to have a relatively low number of Coins,
   so that selection rarely involves all of the distinct Card types.

3) If cost(A) > cost(B), it is usually correct to acquire A instead
   of B when both are available.

4) Penalizing Curse preferences and early game Victory Card preferences
   further reduces the expected number of options during the early game.

5) The [willingness to pass during the Buy Phase](#willingness-to-pass-rather-than-buy) prevents automatic Acquisition of the cheapest and least useful
   Cards—which are precisely those most often among available options.
   Note: Preferences for such Cards can still evolve if they are integral
         to a competent Strategy.


One upshot is that distributions which heavily weight expensive cards relative to
cheap cards while encoding less extreme relative preferences amongst cheap cards
can avoid suffering as badly at the hands of the trade-off between expressivity
and consistency.

Such distributions will reliably select the expensive cards when possible.
If expensive cards are not available, but cheap cards are, the probability
mass of the more expensive cards is proportionally redistributed across
the low cost cards, and the derived distribution will never 'go wrong'
choosing an expensive card instead of the correct cheap card.


### Limitations

#### Scope

Although the Engine supports the entire Base Set, Strategy
evolution is currently limited to the First Game kingdom cards.
And, only 4 Player games are fully supported.


#### Cellar Instructions

For human Agents, the ways to carry out Cellar's instructions
are generated according to the rules. In contrast, computer
Agents only face a heuristic choice between: discarding nothing,
or, discarding all of the Victory cards in their hand, if any.


#### Meta Selection

The naive lookahead heuristic doesn't consider sequences of
Actions or Acquisitions. Since changes to the Supply trigger
the end of the game on the current Turn, but do not end the
Turn early, there is an inherent bias to never consider
Actions which will trigger the end of the game while a
Strategy appears to be losing, but will also enable taking
one or more Actions which will then lead to the Strategy
tying for first or winning outright. The inability to
combine Acquisitions during the Action Phase, during the Buy
Phase, or across both is another blindspot with the same origin.


#### Action Class Preferences

Usually, which Action to play is selected using evolved preferences
over the action class of a Card type. Each Card type has an associated
action class and each Strategy has corresponding preferences—a
distribution over the distinct action classes.

An action class classifies Card types with respect to the nature of their
expected impact on the State when enacted. The three action classes used
here are extremely coarse and based on domain knowledge.

A Card type is either non-terminal or terminal; and, terminals are further
divided into those which do or do not yield card draw. If a Card type has
the non-terminal action class, playing it during the Action Phase will lead
to a net decrease in the number of Action plays permitted to the current
player. For example, the non-terminal Merchant has +1 Action among
its on-play Effects. In contrast, Smithy's only on-play Effect is +3 Cards
so its action class is terminal with card draw.

Assume an Agent never passes the Action Phase unless forced to by the
rules. Then whether the Agent can play multiple Actions consecutively
is determined by: the running total of how many Action plays the Agent
is permitted; and, whether any Action cards are in the Agent's hand.
An Agent starts each Action Phase with a single Action play permitted.

Experimentation and domain knowledge both indicate that—usually—an
Agent should play as many Actions during each Action Phase as possible.
With only this goal in mind, sequencing plays requires a consideration
of both terminality and card draw.

Let H be a strict total order on M, the set of action classes. Then
the following procedure uses H to define a simple policy for Action
selection based on the action classes of available options. First, sort
the Actions by their action class according to H; then, select an Action
with the greatest action class according to H.

For example, let `M := {X, Y, Z}`. One choice for H, `Z > X > Y`, represents
the policy, "Play an Action with action class Z if possible, otherwise
play an Action with action class X if possible, otherwise play an Action
with action class Y".

The action class preferences of each Strategy are identically distributed.
This shared distribution was hand-tuned so that, in conjunction with a
labeling of M, repeatedly using it to select which Action to play
emulates sequencing Action plays according to a strict total order policy
over action classes of the sort just described.

Although the distributions are identical, the labeling of M each Strategy
uses—i.e., which action class maps to X, Y, and Z, respectively—is
subject to natural selection. Of interest is the labeling which corresponds
to the following policy which often leads to maximizing the number of Action
plays per Action Phase:

    Play non-terminals before terminals; and,
    Play terminals with draw before terminals without draw.

Extensions including more robust action class categorization and
the ability to model covariance between acquisition preferences and
action play preferences as a function of the State are obvious next steps.


### Heuristics

#### Automatic Selection

Whenever the set of Consequences generated by a Decision is of size 1,
it is selected automatically and its Effects are resolved.


#### Playing Every Available Treasure

At the start of the Buy Phase, Agents make a choice about which Treasures
to play, and then play them simultaneously before making Purchase decisions.

In the Base Set, there is never a strategic benefit to playing fewer than
the maximum number of possible Treasures. Thus playing every available Treasure
is performed automatically.


#### Victory Point Modifier Avoidance

The genetic algorithm evaluates each Strategy's fitness using its performance in an
Elimination Tournament relative to its peers. Then Strategies which exhibit a
preference for cards which modify Victory Points in the early game—or, Curses in the
endgame—are penalized. This results in a re-ranking of Strategy fitness that biases
the algorithm's search away from early Victory Card acquisition and away from Curses
in general.


#### Endgame Preferences

Strategies have two sets of preferences: one for the early game, and one for the end game.
When the turn with ordinal equal to a Genome's switch index is reached, the associated
Strategy starts using the end game preferences for the remainder of the game.


#### Revealing Moat to Defend Against Attacks

In response to an opponent playing an Attack, Agents have the option to gain immunity
from the attack by first revealing a Moat from their hand. The rules dictate that they
can reveal the Moat as many times as they want in response to the Attack, although
there is no additional benefit after the first reveal. It is difficult to conceive of
scenarios where there would be a strategic benefit to suffering an Attack when a Moat
could be revealed instead. Thus Agents automatically reveal a Moat, once, in response
to any Attack in the Base Set if able.


#### Willingness to Pass Rather than Buy

The method of Purchase selection samples up to `N_PURCHASE_PREFERENCES := 3` distinct Piece
types according to a Strategy's evolved preferences. Each time a distinct Piece is
sampled, any available Purchases are searched in hopes of finding one which will
cause the Agent to gain the sampled Piece. If it is found, it is selected. If no
Purchases are selected by the end of this process, the Strategy passes the Buy Phase,
regardless of how many coins or buys they may have had.

Discretion when Buying is essential for competent play. If Buys were mandatory then
Strategies would quickly clutter their collections with cards which cost 0 (Curses
and Coppers).


#### Dependent Acquisitions

Instructions which read, "Trash X, gain a card from the Supply which satisfies [constraints]"
generate DependentAcquisitions. Rather than prioritizing which Piece the Strategy might want
to trash, selection proceeds by first finding the set of Pieces available across the various
trash choices and choosing one to gain. Then it selects the DependentAcquisition which will
gain that Piece contingent upon trashing a Piece which is preferred the least across the
various trash choices.


#### Meta Selection

If more than 2 piles in the Supply are empty (or the Province pile is empty) at the end of a Turn,
the game ends. Therefore, only an Effect which causes an Agent to gain a card from the Supply can
trigger the end of the game.

A Strategy's select method handles the automatic responses listed above, before dispatching
to methods based on the type of Decision which generated the options being considered.
These methods select options according to the preferences the Strategy has evolved.
The only exception to preference-based selection occurs when the State reaches a point
where, in theory, a single Acquisition could trigger the end of the game. In that case,
meta selection methods are used before the task is delegated to a Decision-specific methods.

Meta selection applies when selecting an Acquisition. Meta selection considers the
State entailed by the resolution of each Acquisition's Effects to determine the hypothetical
rank of the Strategy in a world where they selected that Acquisition. If an Acquisition
would end the game and leave the Strategy in a position where they are winning outright,
it is selected immediately.

Otherwise, if there is an Acquisition which would end the game and leave the Strategy tied
for first place, it is selected. Any other Acquisition which would end the game is removed
from further consideration. If no forced wins or ties are found, then default selection
methods are called.

Meta-meta selection operates during the Action Phase. An Action's action class is a coarse
representation of its impact on the State when played. Strategies normally select which
Action to play according to evolved preferences based on the action class of each available
Action. Meta-meta selection overrides the default approach by automatically selecting any
Actions which are found to generate at least one Acquisition which will force a win or tie
for first (if no wins are available) according to meta selection. If all available Actions
exclusively generate Acquisitions which will end the game with the Strategy losing, then
meta-meta selection passes the Action Phase rather than guarantee itself a loss.


## Future Directions

### Strategy Selection Methods

Support all decisions arising in the Base Set, not just those
generated by the First Game kingdom cards.

In particular, inform deck composition decisions—e.g.,
trashing with Chapel—by deriving a deck evaluation function
from evolved preferences.  Currently, choosing between changes
to one's set of cards which involve different numbers of Pieces
is not supported.

This change will lead naturally to no longer needing to use the
Cellar Heuristic when generating the choices for computer agents.
It will also open the possibility of considering combinations of
Purchases during the Buy Phase.

Determine the minimal addition which will lead to competent
deck sequencing—e.g., which cards to topdeck with Sentry.


### Engine

Add support for Duration cards and for effects which are generated
in response to particular events occuring.

Improve State serialization and support reversible Effects to support
more traditional game tree expansions and associated lookahead heuristics.


### Text Display

Add the necessary data and extend view to work with any Base Set
Kingdom, not just the First Game cards.


### Performance Metrics

Add statistics about the number of cards played/drawn/bought
to the summary of a session. Use these to disambiguate ties
during fitness evaluation based on the typical win condition.


### Preference Generalization

Action sequencing currently relies on the a priori classification
of Actions into three broad categories: non-terminal, terminal with
draw, and terminal without draw. Replace this classification with
one based on an embedding of Actions into an abstract space where
they represent the expected transformation of the State entailed
by their Effects.

With such a representation, preferences evolved for certain Actions
might be generalized to Actions never-before-seen according to
proximity in this transformation space. Adapt Action preferences to
use this generalized representation.


### Switch Index

The switch index is the turn after which Strategies swap to using
their evolved end game preferences rather than their evolved
early game preferences. How its current value was arrived at is
discussed [here](#switch-index).

Instead of a simple turn ordinal rule of thumb, Strategies might
evolve an encoder to represent the State, and swap strategies in
relation to that representation.


## Decisions, Consequences, and Effects

Each Turn is a sequence of Phases which unfold in a pre-defined order.

Each Phase is a loop with its own termination condition and a method
for generating the set of possible moves the current player can
make, given the rules and the State. At the start of each iteration,
the set of possible moves is generated, and the current player selects
which one to make—then makes it.

For example, in the Action Phase, the termination condition is a
conjunction: the current player has not yet elected to pass the
Action Phase this Turn; and, they are entitled to playing at least
1 Action card from their Hand. At the start of the Action Phase,
the set of possible moves includes: passing the Action Phase; and,
playing one of the distinct types of Actions in Hand, if any exist.

When an Agent makes a move, they alter the State. A collection of
distinct changes to the state is a Consequence—and, each component
in such a collection is an Effect. To make a move is to select a
Consequence and then resolve its Effects. The logic which governs
the generation of a set of moves is a Decision.

In the Action Phase, a Decision generates the set of possible moves,
each of which is a Consequence. To make the move of passing the Action
Phase is to select the corresponding Consequence and then resolve its
Effects. In this case, the Consequence contains only one Effect to resolve.
Its resolution transforms the State such that the Action Phase's termination
condition is satisfied.

To play an Action instead of passing is to select that Consequence
instead and resolve its Effects. In this case, there will be a single
Effect, which encapsulates the logic of playing a Piece. Resolving this
Effect will update the State to deincrement the count of permitted
action plays and will move the Piece from the player's hand to their
play area.

Additionally, if the Piece has a Consequence containing Effects to
resolve automatically when it is played—those Effects are resolved;
and, finally, if the Piece has instructions to follow, then those
instructions are followed.

Instructions to follow are represented by Decisions which encode the
logic governing how to generate the legal ways of following those
instructions. Each way of following the instructions can be thought of
as a move available for the player to make. Therefore, to follow
instructions is to select a Consequence representing a distinct way of
following those instructions, and then resolve its Effects.

Although Effects typically transform the State when they are resolved,
some also trigger the resolution of other Effects instead. In this way,
arbitrary sequences of moves with different actors can unfold within
a Phase, as the resolution of one Effect can be nested within the earlier
resolution of a different Effect.
