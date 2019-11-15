import copy
import itertools
from typing import Optional, List, Tuple, Union

import numpy as np
from sortedcontainers import SortedListWithKey

from .tabled_trie import Trie, make_trie


class LevenshteinSearcher:
    """A class to search close words using Levenshtein distance

    Parameters:
        alphabet: the alphabet of characters,
        dictionary: the dictionary of words,
        operation_costs: the Levenshtein operation costs. If it is `None`,
            the basic Levenshtein distance is used.
        allow_spaces: whether spaces are allowed,
        euristics: the lookahead euristics, see [Hulden, 2009] for details
            if ``euristics`` = `None`, no lookahead is applied.
    """

    def __init__(self, alphabet: Union[List[str], str], dictionary: Union[Trie, List[str]],
                 operation_costs: Optional[dict] = None, allow_spaces: bool = False,
                 euristics: Optional[int] = None):
        self.alphabet = alphabet
        self.allow_spaces = allow_spaces
        if isinstance(euristics, int):
            if euristics < 0:
                raise ValueError("Euristics should be non-negative integer or None")
            else:
                self.heuristics = euristics if euristics != 0 else None
        elif euristics is None:
            self.heuristics = None
        else:
            raise ValueError("Euristics should be non-negative integer or None")
        if isinstance(dictionary, Trie):
            self.dictionary = dictionary
        else:
            # converting the dictionary to the trie format
            self.dictionary = make_trie(
                alphabet,
                dictionary,
                make_cashed=True,
                precompute_symbols=self.heuristics,
                allow_spaces=self.allow_spaces,
            )
        self.transducer = SegmentTransducer(alphabet, operation_costs=operation_costs, allow_spaces=allow_spaces)
        self._precompute_heuristics()
        self._define_h_function()

    def __contains__(self, word):
        return word in self.dictionary

    def search(self, word: str, d: int, allow_spaces: bool=True, return_cost: bool=True) -> list:
        """
        Finds all dictionary words in a window of width `d` around `word`
        """
        if not all((c in self.alphabet or (c == " " and self.allow_spaces)) for c in word):
            return []
        return self._trie_search(word, d, allow_spaces=allow_spaces, return_cost=return_cost)

    def _trie_search(self, word: str, d: int, transducer: Optional["SegmentTransducer"]=None,
                     allow_spaces: bool=True, return_cost: bool=True) -> list:
        """
        Finds all `w` with distance(`word`, `w`) <= d using a prefix trie.
        """
        if transducer is None:
            transducer = self.transducer.inverse()
        allow_spaces &= self.allow_spaces
        trie = self.dictionary

        used_agenda_keys = set()
        # priority queue with intermediate results
        agenda = SortedListWithKey(key=(lambda x: x[1]))
        h = self.h_func(word, trie.root)
        key, value = ("", 0, trie.root), (0.0, 0.0, h)
        agenda.add((key, value))
        answer = dict()
        k = 0
        while len(agenda) > 0:
            key, value = agenda.pop(0)
            if key in used_agenda_keys:
                continue
            used_agenda_keys.add(key)
            low, pos, index = key
            cost, g, h = value
            # g --- current value, h --- lower bound of the future cost
            # cost = g + h --- lower bound of the total cost
            k += 1
            max_upperside_length = min(len(word) - pos, transducer.max_up_length)
            for upperside_length in range(max_upperside_length + 1):
                new_pos = pos + upperside_length
                curr_up = word[pos:new_pos]
                if curr_up not in transducer.operation_costs:
                    continue
                for curr_low, curr_cost in transducer.operation_costs[curr_up].items():
                    new_g = g + curr_cost
                    if new_g > d:  # there is no need to calculate `h` if `new_g` is already large.
                        continue
                    if curr_low == " ":
                        if allow_spaces and trie.is_final(index):
                            new_index = trie.root
                        else:
                            new_index = Trie.NO_NODE
                    else:
                        new_index = trie.descend(index, curr_low)
                    if new_index is Trie.NO_NODE:
                        continue
                    new_low = low + curr_low
                    new_h = self.h_func(word[new_pos:], new_index)
                    new_cost = new_g + new_h
                    if new_cost > d:
                        continue
                    new_key = (new_low, new_pos, new_index)
                    new_value = (new_cost, new_g, new_h)
                    if new_pos == len(word) and trie.is_final(new_index):
                        old_g = answer.get(new_low, None)
                        if old_g is None or new_g < old_g:
                            answer[new_low] = new_g
                    agenda.add((new_key, new_value))
        answer = sorted(answer.items(), key=(lambda x: x[1]))
        if return_cost:
            return answer
        else:
            return [elem[0] for elem in answer]

    def _precompute_heuristics(self):
        """
        Calculates future symbols and operation costs for h-heuristic (Hulden, 2009)
        """
        if self.heuristics is None:
            return
        # calculates minimal costs of symbol removal and insertion
        removal_costs = {a: np.inf for a in self.alphabet}
        insertion_costs = {a: np.inf for a in self.alphabet}
        if self.allow_spaces:
            removal_costs[" "] = np.inf
            insertion_costs[" "] = np.inf
        for up, costs in self.transducer.operation_costs.items():
            for low, cost in costs.items():
                if up == low:
                    continue
                if up != "":
                    removal_cost = cost / len(up)
                    for a in up:
                        removal_costs[a] = min(removal_costs[a], removal_cost)
                if low != "":
                    insertion_cost = cost / len(low)
                    for a in low:
                        insertion_costs[a] = min(insertion_costs[a], insertion_cost)
        # costs for symbols absence
        self._absense_costs_by_node = _precompute_absense_costs(
            self.dictionary, removal_costs, insertion_costs, self.heuristics, self.allow_spaces
        )
        self._temporary_heuristics = [dict() for i in range(len(self.dictionary))]

    def _define_h_function(self):
        if self.heuristics in [None, 0]:
            self.h_func = lambda *x: 0.0
        else:
            self.h_func = self._euristic_h_function

    def _euristic_h_function(self, suffix: str, index: int) -> float:
        """
        Computes h-heuristics (Hulden, 2009) for current dictionary vertex

        Args:
            suffix: unread suffix of input word
            index: current dictionary node index
        Returns:
            cost: lower bound for the cost for attaching ``suffix`` to ``index`` node
        """
        if self.heuristics > 0:
            suffix = suffix[: self.heuristics]
        # caching results
        index_temporary_euristics = self._temporary_heuristics[index]
        cost = index_temporary_euristics.get(suffix, None)
        if cost is not None:
            return cost
        absense_costs = self._absense_costs_by_node[index]
        costs = np.zeros(dtype=np.float64, shape=(self.heuristics,))
        # costs[j] --- cost estimate for looking ahead for j steps
        for i, a in enumerate(suffix):
            costs[i:] += absense_costs[a][i:]
        cost = max(costs)
        index_temporary_euristics[suffix] = cost
        return cost

    def _minimal_replacement_cost(self, first, second):
        first_symbols, second_symbols = set(), set()
        removal_cost, insertion_cost = 0, 0
        for a, b in itertools.zip_longest(first, second, fillvalue=None):
            if a is not None:
                first_symbols.add(a)
            if b is not None:
                second_symbols.add(b)
            removal_cost = max(removal_cost, len(first_symbols - second_symbols))
            insertion_cost = max(insertion_cost, len(second_symbols - first_symbols))
        return min(removal_cost, insertion_cost)


def _precompute_absense_costs(dictionary: Trie, removal_costs: dict,
                              insertion_costs: dict, n: int, allow_spaces: bool=False) -> List[dict]:
    """
    Calculates the minimum cost of symbol insertion and removal for all nodes in dictionary

    Args:
        dictionary: dictionary stored as an acyclic automaton,
        removal_costs: symbol removal costs,
        insertion_costs: symbol insertion costs
        n: lookahead depth

    Returns:
        answer: the list of symbol costs for each node of the dictionary
            answer[i][a][j] is minimal cost for symbol `a`
            to appear as `j`-th symbol after the trie node `i`
    """
    answer = [dict() for node in dictionary.data]
    if n == 0:
        return answer
    curr_alphabet = copy.copy(dictionary.alphabet)
    if allow_spaces:
        curr_alphabet += [" "]
    for l, (costs_in_node, node) in enumerate(zip(answer, dictionary.data)):
        # minimal removal costs
        curr_node_removal_costs = np.empty(dtype=np.float64, shape=(n,))
        if len(node[0]) > 0:
            curr_node_removal_costs[0] = min(removal_costs[symbol] for symbol in node[0])
            for j, symbols in enumerate(node[1:], 1):
                if len(symbols) == 0:
                    curr_node_removal_costs[j:] = curr_node_removal_costs[j - 1]
                    break
                curr_cost = min(removal_costs[symbol] for symbol in symbols)
                curr_node_removal_costs[j] = min(curr_node_removal_costs[j - 1], curr_cost)
        else:
            curr_node_removal_costs[:] = np.inf
        # minimal insertion costs
        for a in curr_alphabet:
            curr_symbol_costs = np.empty(dtype=np.float64, shape=(n,))
            curr_symbol_costs.fill(insertion_costs[a])
            for j, symbols in enumerate(node):
                if a in symbols:
                    curr_symbol_costs[j:] = 0.0
                    break
                curr_symbol_costs[j] = min(curr_symbol_costs[j], curr_node_removal_costs[j])
            costs_in_node[a] = curr_symbol_costs
    return answer


class SegmentTransducer:
    """
    Our implementation of weighted finite transducer.
    The transducer performs substitutions from a given set of operations.

    Args:
        alphabet: alphabet
        operation_costs: dictionary of the form {(`up`, `low`): `cost`},
            where `up` and `low` are the components of substitution operation.
        allow_spaces:
            whether spaces are allowed in transduction components.
            It is used only with default operation costs (``operation_costs``=``None``).
    """

    def __init__(self, alphabet: Union[List, str],
                 operation_costs: Optional[dict]=None, allow_spaces: bool=False):
        self.alphabet = alphabet
        if operation_costs is None:
            self._make_default_operation_costs(allow_spaces=allow_spaces)
        elif not isinstance(operation_costs, dict):
            raise TypeError("Operation costs must be a dictionary")
        else:
            self.operation_costs = operation_costs
        self._make_reversed_operation_costs()
        self._make_maximal_key_lengths()

    def get_operation_cost(self, up: str, low: str) -> float:
        """
        Returns the cost of elementary transduction `up`->`low`
        or np.inf if there is no such elementary transduction

        Args:
            up, low: upper and lower parts of the transduction (substitution)

        Returns:
            cost: cost of the transduction `up`->`low`
                (`np.inf` if there is no such transduction)
        """
        up_costs = self.operation_costs.get(up, None)
        if up_costs is None:
            return np.inf
        cost = up_costs.get(low, np.inf)
        return cost

    def inverse(self):
        """
        Inverts current transducer.
        """
        inversed_transducer = SegmentTransducer(self.alphabet, operation_costs=dict())
        inversed_transducer.operation_costs = self._reversed_operation_costs
        inversed_transducer._reversed_operation_costs = self.operation_costs
        inversed_transducer.max_low_length = self.max_up_length
        inversed_transducer.max_up_length = self.max_low_length
        inversed_transducer.max_low_lengths_by_up = self.max_up_lengths_by_low
        inversed_transducer.max_up_lengths_by_low = self.max_low_lengths_by_up
        return inversed_transducer

    def distance(self, first: str, second: str,
                 return_transduction: bool = False) -> Union[Tuple[float, List], float]:
        """
        Calculates the cost of replacing ``first`` by ``second``.

        Args:
            first:
            second:
                Upper and lower transduction elements

            return_transduction: whether to return the corresponding transduction

        Returns: minimal cost of the transduction that converts `first` to `second`.
            If ``return_transduction``=True, additionally returns the transduction itself.
        """
        if return_transduction:
            add_pred = lambda x, y: (y == np.inf or x < y)
        else:
            add_pred = lambda x, y: (y == np.inf or x <= y)
        clear_pred = lambda x, y: x < y < np.inf
        update_func = lambda x, y: min(x, y)
        costs, backtraces = self._fill_levenshtein_table(first, second, update_func, add_pred, clear_pred)
        final_cost = costs[-1][-1]
        if final_cost == np.inf:
            transductions = [None]
        elif return_transduction:
            transductions = self._backtraces_to_transductions(
                first, second, backtraces, final_cost, return_cost=False)
        if return_transduction:
            return final_cost, transductions
        else:
            return final_cost

    def transduce(self, first: str, second: str, threshold: float) -> List[Tuple]:
        """
        Returns all transductions that convert `first` to `second` and cost less than `threshold`.

        Returns:
            result: the list of pairs (transduction, cost) for all such transductions.
        """
        add_pred = lambda x, y: x <= threshold
        clear_pred = lambda x, y: False
        update_func = lambda x, y: min(x, y)
        costs, backtraces = self._fill_levenshtein_table(
            first, second, update_func, add_pred, clear_pred, threshold=threshold
        )
        result = self._backtraces_to_transductions(first, second, backtraces, threshold, return_cost=True)
        return result

    def lower_transductions(self, word: str, max_cost: float, return_cost: bool=True):
        """
        Returns all transductions whose upper element is `word`
        and whose cost does not exceed `max_cost`.

        Returns:
            result:
                a list of all pairs (transduction, cost) if return_cost = True,
                a list of all such transductions if return_cost = False.
                The list is sorted from cheaper transduction to more expensive.
        """
        prefixes = [[] for i in range(len(word) + 1)]
        prefixes[0].append(((), 0.0))
        for pos in range(len(prefixes)):
            prefixes[pos] = self._perform_insertions(prefixes[pos], max_cost)
            max_upperside_length = min(len(word) - pos, self.max_up_length)
            for upperside_length in range(1, max_upperside_length + 1):
                up = word[pos : pos + upperside_length]
                for low, low_cost in self.operation_costs.get(up, dict()).items():
                    for transduction, cost in prefixes[pos]:
                        new_cost = cost + low_cost
                        if new_cost <= max_cost:
                            new_transduction = transduction + (up, low)
                            prefixes[pos + upperside_length].append((new_transduction, new_cost))
        answer = sorted(prefixes[-1], key=(lambda x: x[0]))
        if return_cost:
            return answer
        else:
            return [elem[0] for elem in answer]

    def lower(self, word: str, max_cost: float, return_cost: bool=True):
        """
        Returns lower elements for all transductions
        whose upper element is `word` and whose cost does not exceed `max_cost`.

        Returns:
            result:
                a list of all pairs (transduction, cost) if return_cost = True,
                a list of all such transductions if return_cost = False.
                The list is sorted from cheaper transduction to more expensive.
        """
        transductions = self.lower_transductions(word, max_cost, return_cost=True)
        answer = dict()
        for transduction, cost in transductions:
            low = "".join(elem[1] for elem in transductions)
            curr_cost = answer.get(low, None)
            if curr_cost is None or cost < curr_cost:
                answer[low] = cost
        answer = sorted(answer.items(), key=(lambda x: x[1]))
        if return_cost:
            return answer
        else:
            return [elem[0] for elem in answer]

    def upper(self, word: str, max_cost: float, return_cost: bool=True):
        """
        Returns upper elements for all transductions
        whose lower element is `word` and whose cost does not exceed `max_cost`.

        Returns:
            result:
                a list of all pairs (transduction, cost) if return_cost = True,
                a list of all such transductions if return_cost = False.
                The list is sorted from cheaper transduction to more expensive.
        """
        inversed_transducer = self.inverse()
        return inversed_transducer.lower(word, max_cost, return_cost)

    def upper_transductions(self, word: str, max_cost: float, return_cost: bool=True):
        """
        Returns all transductions whose lower element is `word`
        and whose cost does not exceed `max_cost`.

        Returns:
            result:
                a list of all pairs (transduction, cost) if return_cost = True,
                a list of all such transductions if return_cost = False.
                The list is sorted from cheaper transduction to more expensive.
        """
        inversed_transducer = self.inverse()
        return inversed_transducer.lower_transductions(word, max_cost, return_cost)

    def _fill_levenshtein_table(self, first: str, second: str, update_func: callable,
                                add_pred: callable, clear_pred: callable,
                                threshold: float=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamically calculates the table of costs
        for transduction between `first` and `second`.
        costs[i][j] --- the minimum cost of transduction between first[:i] and second[:j].

        Args:
            first, second: upper and lower transduction elements
            update_func: a callable `update_func`(x, y),
                that returns the new value of the table cell
                for the old value `y`, and new candidate value `x`.
            add_pred: a callable `add_pred`(x, y),
                add_pred(x, y) returns whether we shall add a new element
                to current backtraces cell if `x` is the new cost of the
                current table cell add `y` was the old one.
            clear_pred: a callable `clear_pred`(x, y),
                clear_pred(x, y) returns whether we shall clear
                current backtraces cell if `x` is the new cost of the
                current table cell add `y` was the old one.

        Returns:
            costs: an array of floats of shape (`len`(`first`) +1, `len`(`second`) +1)
                costs[i,j] contains the distance between first [:i] and second [:j].
            backtraces: an array of lists of shape (`len`(`first`) +1, `len`(`second`) +1)
                backtraces[i,j] contains the links to the previous cells in the paths
                correspondings to optimal transductions between first [:i] and second [:j].
        """
        m, n = len(first), len(second)
        # if `threshold`=`None`, the threshold equals double identity transduction cost.
        if threshold is None:
            threshold = 0.0
            for a, b in zip(first, second):
                threshold += self.get_operation_cost(a, b)
            if m > n:
                for a in first[n:]:
                    threshold += self.get_operation_cost(a, "")
            elif m < n:
                for b in second[m:]:
                    threshold += self.get_operation_cost("", b)
            threshold *= 2
        # initialization of output arrays
        costs = np.zeros(shape=(m + 1, n + 1), dtype=np.float64)
        costs[:] = np.inf
        backtraces = [None] * (m + 1)
        for i in range(m + 1):
            backtraces[i] = [[] for j in range(n + 1)]
        costs[0][0] = 0.0
        for i in range(m + 1):
            for i_right in range(i, min(i + self.max_up_length, m) + 1):
                up = first[i:i_right]
                max_low_length = self.max_low_lengths_by_up.get(up, -1)
                if max_low_length == -1:  # no up key in transduction
                    continue
                up_costs = self.operation_costs[up]
                for j in range(n + 1):
                    if costs[i][j] > threshold:
                        continue
                    if len(backtraces[i][j]) == 0 and i + j > 0:
                        continue  # no backlinks found
                    for j_right in range((j if i_right > i else j + 1), min(j + max_low_length, n) + 1):
                        low = second[j:j_right]
                        curr_cost = up_costs.get(low, np.inf)
                        old_cost = costs[i_right][j_right]
                        new_cost = costs[i][j] + curr_cost
                        if new_cost > threshold:
                            continue
                        if add_pred(new_cost, old_cost):
                            if clear_pred(new_cost, old_cost):
                                backtraces[i_right][j_right] = []
                            costs[i_right][j_right] = update_func(new_cost, old_cost)
                            backtraces[i_right][j_right].append((i, j))
        return costs, backtraces

    def _make_reversed_operation_costs(self):
        """
        Fills the ``_reversed_operation_costs`` array
        using the available ``operation_costs`` array
        """
        _reversed_operation_costs = dict()
        for up, costs in self.operation_costs.items():
            for low, cost in costs.items():
                if low not in _reversed_operation_costs:
                    _reversed_operation_costs[low] = dict()
                _reversed_operation_costs[low][up] = cost
        self._reversed_operation_costs = _reversed_operation_costs

    def _make_maximal_key_lengths(self):
        """
        Calculates the dictionaries that contain the maximal length
        of the lower transduction part for each upper transduction part,
        as well as calculates maximal upper part lengths
        for given lower transduction parts.
        """
        self.max_up_length = (max(len(up) for up in self.operation_costs)
                              if len(self.operation_costs) > 0 else -1)
        self.max_low_length = (
            max(len(low) for low in self._reversed_operation_costs)
            if len(self._reversed_operation_costs) > 0 else -1
        )
        self.max_low_lengths_by_up, self.max_up_lengths_by_low = dict(), dict()
        for up, costs in self.operation_costs.items():
            self.max_low_lengths_by_up[up] = max(len(low) for low in costs) if len(costs) > 0 else -1
        for low, costs in self._reversed_operation_costs.items():
            self.max_up_lengths_by_low[low] = max(len(up) for up in costs) if len(costs) > 0 else -1

    def _backtraces_to_transductions(self, first: str, second: str, backtraces: np.ndarray,
                                     threshold: float, return_cost: bool=False) -> List:
        """
        Recovers the transduction using the backtraces table.

        Args:
            first, second: upper and lower transduction parts
            backtraces: the backtraces matrix
            threshold: the maximal allowed cost of the transduction
            return_cost: if `True`, then transduction costs are returned

        Returns:
            result: the list of output transductions whose cost does not exceed ``threshold``.
                If ``return_cost``=`True`, transduction costs are also returned.
        """
        m, n = len(first), len(second)
        agenda = [None] * (m + 1)
        for i in range(m + 1):
            agenda[i] = [[] for j in range(n + 1)]
        agenda[m][n] = [((), 0.0)]
        for i_right in range(m, -1, -1):
            for j_right in range(n, -1, -1):
                current_agenda = agenda[i_right][j_right]
                if len(current_agenda) == 0:
                    continue
                for (i, j) in backtraces[i_right][j_right]:
                    up, low = first[i:i_right], second[j:j_right]
                    add_cost = self.operation_costs[up][low]
                    for elem, cost in current_agenda:
                        new_cost = cost + add_cost
                        if new_cost <= threshold:  # high cost transduction removal
                            agenda[i][j].append((((up, low),) + elem, new_cost))
        if return_cost:
            return agenda[0][0]
        else:
            return [elem[0] for elem in agenda[0][0]]

    def _perform_insertions(self, initial: List[Tuple], max_cost: float) -> List:
        """
        returns all transductions whose value does not exceed `max_cost`,
        that can be obtained from ``initial`` by appending
        transductions with empty upper part.

        Args:
            initial: list of initial transductions of the form (transduction, cost)
            max_cost: maximal allowed cost

        Returns:
            final: the final list of transductions of the form (transduction, cost)
        """
        queue = list(initial)
        final = initial
        while len(queue) > 0:
            transduction, cost = queue[0]
            queue = queue[1:]
            for string, string_cost in self.operation_costs[""].items():
                new_cost = cost + string_cost
                if new_cost <= max_cost:
                    new_transduction = transduction + ("", string)
                    final.append((new_transduction, new_cost))
                    queue.append((new_transduction, new_cost))
        return final

    def _make_default_operation_costs(self, allow_spaces=False):
        """
        sets 1.0 cost for every replacement, insertion, deletion and transposition
        """
        self.operation_costs = dict()
        self.operation_costs[""] = {c: 1.0 for c in list(self.alphabet) + [" "]}
        for a in self.alphabet:
            current_costs = {c: 1.0 for c in self.alphabet}
            current_costs[a] = 0.0
            current_costs[""] = 1.0
            if allow_spaces:
                current_costs[" "] = 1.0
            self.operation_costs[a] = current_costs
        # transposition
        for a, b in itertools.permutations(self.alphabet, 2):
            self.operation_costs[a + b] = {b + a: 1.0}
        # spaces
        if allow_spaces:
            self.operation_costs[" "] = {c: 1.0 for c in self.alphabet}
            self.operation_costs[" "][""] = 1.0
