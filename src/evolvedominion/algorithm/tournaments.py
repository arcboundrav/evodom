from collections import Counter
from itertools import chain, cycle
from evolvedominion.params import GROUPSIZE
from evolvedominion.engine.session import Session


class Tournament:
    __slots__ = (
        "N",
        "k",
        "entrants",
        "entrant_scores",
        "n_rounds",
        "current_round_index",
        "sessions",
    )
    def __init__(self, N):
        self.N = N
        self.k = GROUPSIZE
        self.entrants = list()
        self.entrant_scores = Counter()
        self.n_rounds = 1
        self.current_round_index = 0
        self.sessions = self.init_sessions()

    def init_sessions(self):
        sessions = []
        for i in range((self.N // self.k)):
            sessions.append(Session())
        return sessions

    def accept_entrants(self, entrants):
        self.entrants = entrants
        self.current_round_index = 0
        self.entrant_scores.clear()

    def play(self, session):
        session.play()
        return {player.pid:player.score for player in session.players}

    def assign_entrants_to_sessions(self):
        raise NotImplementedError

    def rank_entrants(self):
        raise NotImplementedError

    def _run(self, pool):
        raise NotImplementedError

    def run(self, entrants, pool):
        self.accept_entrants(entrants)
        self._run(pool)


class Seeding(Tournament):
    """
    Establish seeds for an Elimination tournament by playing Dominion and
    ranking entrants based on how they place.
    """
    def __init__(self, N):
        super().__init__(N=N)

    def assign_entrants_to_sessions(self):
        s = 0
        for i in range(0, self.N, self.k):
            self.sessions[s].accept_players(self.entrants[i:(i+self.k)])
            s = s + 1

    def rank_entrants(self):
        self.entrants.sort(key=lambda entrant: (self.entrant_scores[entrant.pid]), reverse=True)
        for i in range(self.N):
            self.entrants[i].phenotype.genome.rank = i

    def _run(self, pool):
        self.assign_entrants_to_sessions()
        for session_result in pool.imap_unordered(func=self.play, iterable=self.sessions, chunksize=self.k):
            self.entrant_scores.update(session_result)
        self.rank_entrants()


class Elimination(Tournament):
    __slots__ = (
        "entrant_map",
        "losers",
        "n_losers",
        "dropout",
        "sessions_to_use",
    )
    def __init__(self, N):
        super().__init__(N=N)
        self.entrant_map = {}
        self.losers = list()
        self.n_losers = self.N - 1
        self.n_rounds = 0
        self.dropout = list()
        self.sessions_to_use = list()
        self.solve_schedule()

    def accept_entrants(self, entrants):
        super().accept_entrants(entrants)
        self.entrant_map = {entrant.pid:entrant for entrant in entrants}
        self.losers = []
        self.n_losers = self.N - 1

    def solve_schedule(self):
        """
        Solve the number of rounds this tournament will take; and, solve how many entrants to
        eliminate during each of these rounds, given N total entrants and k entrants per game.
        The number of entrants eliminated each round should be as close as possible to the ideal
        of only 1 entrant per game being eliminated, while ensuring that:
            A) All entrants are eliminated after the final round; and,
            B) During every round, all remaining entrants are assigned to a game with k players.
        """
        n = self.N
        while (n > self.k):
            n_participants_to_eliminate = (n // self.k) + ((n - (n // self.k)) % self.k)
            self.dropout.append(n_participants_to_eliminate)
            n = n - n_participants_to_eliminate
            self.n_rounds = self.n_rounds + 1
        self.dropout.append(self.k)

    def assign_entrants_to_sessions(self, participant_pids):
        """
        Slight variation on the typical scheduling of a single elimination tournament.
        For the first round the pids are sorted according to the seeds assigned by
        the Seeding tournament. For each other round, they are sorted according to how
        well the corresponding entrant performed in the immediately preceding round.
        """
        n_participants = len(participant_pids)
        half_n_participants = n_participants // 2
        n_groups = n_participants // self.k
        groups = [[] for i in range(n_groups)]
        group_indices = cycle(range(n_groups))
        reversed_group_indices = cycle(reversed(range(n_groups)))

        for i in range(half_n_participants):
            group_index = next(group_indices)
            participant_pid = participant_pids.pop()
            participant = self.entrant_map[participant_pid]
            groups[group_index].append(participant)

        group_indices = cycle(reversed(range(n_groups)))
        for i in range(half_n_participants):
            group_index = next(group_indices)
            participant_pid = participant_pids.pop()
            participant = self.entrant_map[participant_pid]
            groups[group_index].append(participant)

        for i in range(n_groups):
            self.sessions[i].accept_players(groups[i])
        self.sessions_to_use = self.sessions[:n_groups]

    def solve_n_to_keep(self, n_participants):
        return n_participants - self.dropout[self.current_round_index]

    def eliminate_entrants_by_pid(self, pids):
        """
        pids is sorted in descending order in terms of performance in the most
        recent round of the tournament. Since self.losers is constructed by
        appending to a list, and a lower rank is better, reverse pids before
        assigning ranks.
        """
        for pid in reversed(pids):
            eliminated_entrant = self.entrant_map[pid]
            eliminated_entrant.phenotype.genome.rank = self.n_losers
            self.losers.append(pid)
            self.n_losers = self.n_losers - 1

    def rank_entrants(self):
        """
        self.losers is constructed by appending the pid of each entrant in the order
        they were eliminated from the tournament. The genomes of the entrants were
        assigned ranks during this process. The simulation expects self.entrants to
        be sorted in ascending order in terms of rank---i.e., self.entrants[0] won
        the tournament.
        """
        self.entrants = list(self.entrant_map[key] for key in reversed(self.losers))

    def _run(self, pool):
        participant_pids = list(entrant.pid for entrant in self.entrants)
        n_participants = len(participant_pids)
        while (self.current_round_index < self.n_rounds):
            n_participants = self.solve_n_to_keep(n_participants)
            self.entrant_scores.clear()
            self.assign_entrants_to_sessions(participant_pids)
            for session_result in pool.imap_unordered(func=self.play,
                                                      iterable=self.sessions_to_use,
                                                      chunksize=self.k):
                self.entrant_scores.update(session_result)
            sorted_pids = (t[0] for t in sorted(self.entrant_scores.items(), key=lambda t: (t[1]), reverse=True))
            participant_pids.clear()
            for i in range(n_participants):
                participant_pids.append(next(sorted_pids))
            remaining_pids = list(sorted_pids)
            self.eliminate_entrants_by_pid(remaining_pids)
            self.current_round_index = self.current_round_index + 1
        self.rank_entrants()
