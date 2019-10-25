import copy
import itertools

import numpy as np
from sortedcontainers import SortedListWithKey

from .tabled_trie import Trie, make_trie


class LevenshteinSearcher:
    """
     A class for finding related words
     according to Levenshtein distance

    """

    def __init__(self, alphabet, dictionary, operation_costs=None, allow_spaces=False, euristics="none"):
        self.alphabet = alphabet
        self.allow_spaces = allow_spaces
        if isinstance(euristics, int):
            if euristics < 0:
                raise ValueError("Euristics should be non-negative integer or None")
            else:
                self.euristics = euristics if euristics != 0 else None
        elif euristics in ["none", "None", None]:
            self.euristics = None
        else:
            raise ValueError("Euristics should be non-negative integer or None")
        if isinstance(dictionary, Trie):
            # the dictionary is already transferred in the form of forest
            self.dictionary = dictionary
        else:
            self.dictionary = make_trie(
                alphabet,
                dictionary,
                make_cashed=True,
                precompute_symbols=self.euristics,
                allow_spaces=self.allow_spaces,
            )
        self.transducer = SegmentTransducer(alphabet, operation_costs=operation_costs, allow_spaces=allow_spaces)
        self._precompute_euristics()
        self._define_h_function()

    def __contains__(self, word):
        return word in self.dictionary

    def search(self, word, d, allow_spaces=True, return_cost=True):
        """
        Finds all dictionary words in d-window from word
        """
        if not all((c in self.alphabet or (c == " " and self.allow_spaces)) for c in word):
            return []
            # raise ValueError("{0} contains an incorrect symbol".format(word))
        return self._trie_search(word, d, allow_spaces=allow_spaces, return_cost=return_cost)

    def _trie_search(self, word, d, transducer=None, allow_spaces=True, return_cost=True):
        """
        Finds all words in the prefix forest, the distance to which
         in accordance with a given converter does not exceed d
        """
        if transducer is None:
            #  TODO: check spaces
            transducer = self.transducer.inverse()
        allow_spaces &= self.allow_spaces
        trie = self.dictionary
        #  variable initialization
        used_agenda_keys = set()
        agenda = SortedListWithKey(key=(lambda x: x[1]))
        h = self.h_func(word, trie.root)
        # agenda[self.agenda_key("", 0, trie.root)] = (0.0, 0.0, h)
        key, value = ("", 0, trie.root), (0.0, 0.0, h)
        agenda.add((key, value))
        answer = dict()
        k = 0
        # priority queue with intermediate results
        while len(agenda) > 0:
            key, value = agenda.pop(0)
            if key in used_agenda_keys:
                continue
            used_agenda_keys.add(key)
            low, pos, index = key
            cost, g, h = value
            # g --- current value, h --- lower estimate of future value
            # cost = g + h --- lower total cost estimate
            k += 1
            max_upperside_length = min(len(word) - pos, transducer.max_up_length)
            for upperside_length in range(max_upperside_length + 1):
                new_pos = pos + upperside_length
                curr_up = word[pos:new_pos]
                if curr_up not in transducer.operation_costs:
                    continue
                for curr_low, curr_cost in transducer.operation_costs[curr_up].items():
                    new_g = g + curr_cost
                    if new_g > d:  # h cannot be calculated if g > d
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

    def _precompute_euristics(self):
        """
        Calculates future symbols and costs of operations 
        for h-heuristics
        """
        if self.euristics is None:
            return
        # calculation of the minimum cost of the operation,
        # leading to the appearance ('-') of this character
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
        # Calculating possible future characters in tree nodes
        # precompute_future_symbols (self.dictionary, self.euristics, self.allow_spaces)
        # calculating symbol loss costs in tree nodes
        self._absense_costs_by_node = _precompute_absense_costs(
            self.dictionary, removal_costs, insertion_costs, self.euristics, self.allow_spaces
        )
        # array to save heuristics
        self._temporary_euristics = [dict() for i in range(len(self.dictionary))]

    def _define_h_function(self):
        if self.euristics in [None, 0]:
            self.h_func = lambda *x: 0.0
        else:
            self.h_func = self._euristic_h_function

    def _euristic_h_function(self, suffix, index):
        """
        Computing h-heuristics from Hulden, 2009 for the current dictionary top

        Arguments:
        ----------
        suffix: string
            unread suffix of input word
        index: int
            index of the current node in the dictionary
        Returns:
        -----------
        cost: float
            lower estimate for replacement cost,
            leading to the `suffix` suffix,
            if the read prefix of the word without a typo
            brought to the top with `index` number
        """
        if self.euristics > 0:
            suffix = suffix[: self.euristics]
        # caching results
        index_temporary_euristics = self._temporary_euristics[index]
        cost = index_temporary_euristics.get(suffix, None)
        if cost is not None:
            return cost
        # extraction of the necessary data from arrays
        absense_costs = self._absense_costs_by_node[index]
        data = self.dictionary.data[index]
        costs = np.zeros(dtype=np.float64, shape=(self.euristics,))
        # costs[j] --- penalty estimate when looking ahead by j characters
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


def _precompute_absense_costs(dictionary, removal_costs, insertion_costs, n, allow_spaces=False):
    """
    Calculates the minimum cost for a new character to appear in dictionary nodes
    according to fines from costs

    Arguments:
    ---------------
    dictionary: Trie
        dictionary stored as an acyclic automaton

    removal_costs: dict
        character removal fines

    insertion_costs: dict
        character insertion fines

    n: int
        depth of `` looking ahead '' in the dictionary

    Returns
    ---------------
    answer: list of dicts, len(answer) = len(dictionary)
        answer [i] [a] [j] is equal to the minimum penalty for the appearance of the character a
        at the j-th position at the vertex number i
    """
    answer = [dict() for node in dictionary.data]
    if n == 0:
        return answer
    curr_alphabet = copy.copy(dictionary.alphabet)
    if allow_spaces:
        curr_alphabet += [" "]
    for l, (costs_in_node, node) in enumerate(zip(answer, dictionary.data)):
        # determination of the minimum cost of removing characters
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
        # determination of the minimum cost of insertion
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
    A class that implements a weighted final transducer,
    performing replacements from a given list of operations

    Arguments:
    ----------
    alphabet: list
        alphabet

    operation_costs: dict or None (optional, default = None)
        dictionary like {(up, low): cost}

    allow_spaces: bool (optional, default = False)
        whether transduction elements containing a space are allowed
        (used only if operation costs are not explicitly set
        and they are equal to the default value)
    """

    def __init__(self, alphabet, operation_costs=None, allow_spaces=False):
        self.alphabet = alphabet
        if operation_costs is None:
            self._make_default_operation_costs(allow_spaces=allow_spaces)
        elif not isinstance(operation_costs, dict):
            raise TypeError("Operation costs must be a dictionary")
        else:
            self.operation_costs = operation_costs
        self._make_reversed_operation_costs()
        self._make_maximal_key_lengths()
        # self.maximal_value_lengths = {}
        # for up, probs in self.operation_costs.items():
        # TOO MUCH CALLS NEEDS TO REMEMBER
        # MAXIMUM KEY LENGTHS WHEN HANDLING
        # max_low_length = max(len(low) for low in probs) if (len(probs) > 0) else -1
        # self.maximal_value_lengths[up] = self.maximal_key_length

    def get_operation_cost(self, up, low):
        """
        Returns the cost of elemental transduction up-> low
        or np.inf if there is no such elementary transduction

        Arguments:
        ----------
        up, low: string
            elemental transduction elements

        Returns:
        -----------
        cost: float
            elementary transduction cost up-> low
            (np.inf if such transduction is absent)
        """
        up_costs = self.operation_costs.get(up, None)
        if up_costs is None:
            return np.inf
        cost = up_costs.get(low, np.inf)
        return cost

    def inverse(self):
        """
        Build a transformer specifying the inverse final transformation
        """
        # SIMPLIFY THE CALL!!!
        inversed_transducer = SegmentTransducer(self.alphabet, operation_costs=dict())
        inversed_transducer.operation_costs = self._reversed_operation_costs
        inversed_transducer._reversed_operation_costs = self.operation_costs
        inversed_transducer.max_low_length = self.max_up_length
        inversed_transducer.max_up_length = self.max_low_length
        inversed_transducer.max_low_lengths_by_up = self.max_up_lengths_by_low
        inversed_transducer.max_up_lengths_by_low = self.max_low_lengths_by_up
        return inversed_transducer

    def distance(self, first, second, return_transduction=False):
        """
        Calculates the minimum cost transduction,
        mapping `first` to `second`

        Arguments:
        -----------
        first: string
        second: string
            Upper and lower transduction elements

        return_transduction: bool (optional, default = False)
            should minimum weight transduction be returned
            (see return value)

        Returns:
        -----------
        (final_cost, transductions): tuple (float, list)
            if return_transduction = True, then returns
            minimum cost of transduction converting first to second
            and a list of transductions with a given value

        final_cost: float
            if return_transduction = False, then returns
            minimum cost of transduction converting first to second
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
            transductions = self._backtraces_to_transductions(first, second, backtraces, final_cost, return_cost=False)
        if return_transduction:
            return final_cost, transductions
        else:
            return final_cost

    def transduce(self, first, second, threshold):
        """
        Returns all transductions translating first to second,
        whose value does not exceed threshold

        Returns:
        ----------
        result: list
            type list [(transduction, cost)]
        """
        add_pred = lambda x, y: x <= threshold
        clear_pred = lambda x, y: False
        update_func = lambda x, y: min(x, y)
        costs, backtraces = self._fill_levenshtein_table(
            first, second, update_func, add_pred, clear_pred, threshold=threshold
        )
        result = self._backtraces_to_transductions(first, second, backtraces, threshold, return_cost=True)
        return result

    def lower_transductions(self, word, max_cost, return_cost=True):
        """
        Returns all transductions with the top word element,
        whose value does not exceed max_cost

         Returns:
        ----------
        result: list
            a list of the form [(transduction, cost)] if return_cost = True
            transduction list if return_cost = False
            the list is sorted in ascending order of transduction cost
        """
        prefixes = [[] for i in range(len(word) + 1)]
        prefixes[0].append(((), 0.0))
        for pos in range(len(prefixes)):
            # insertions
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

    def lower(self, word, max_cost, return_cost=True):
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

    def upper(self, word, max_cost, return_cost=True):
        inversed_transducer = self.inverse()
        return inversed_transducer.lower(word, max_cost, return_cost)

    def upper_transductions(self, word, max_cost, return_cost=True):
        inversed_transducer = self.inverse()
        return inversed_transducer.lower_transductions(word, max_cost, return_cost)

    def _fill_levenshtein_table(self, first, second, update_func, add_pred, clear_pred, threshold=None):
        """
        A function that dynamically populates the transduction cost table
        costs [i] [j] --- the minimum cost of transduction,
        translating first [: i] to second [: j]

        Arguments:
        ----------
        first, second: string
            Upper and lower transduction elements
        update_func: callable, float * float -> bool
            update_func (x, y) returns the new value in the table cell costs,
            if the old value is y, and the potentially new value is x
            everywhere update_func = min
        add_pred: callable: float * float -> bool
            add_pred (x, y) returns whether adding
            a new element p of cost x to the cell backtraces [i] [j]
            depending on the value of costs [i] [j] = y and current value x
        clear_pred: callable: float * float -> bool
            clear_pred (x, y) returns whether cleaning is performed
            cells backtraces [i] [j] depending on the value of costs [i] [j] = y
            and the current value x of the element p being added to this cell

        Returns:
        -----------
        costs: array, dtype = float, shape = (len (first) +1, len (second) +1)
            an array in the cell with indices i whose j is stored
            minimum cost of transduction converting first [: i] to second [: j]
        backtraces: array, dtype = list, shape = (len (first) +1, len (second) +1)
            an array in the cell with indices i, j of which are stored
            backlinks to the previous cell in optimal transduction,
            cell leading backtraces [i] [j]
        """
        m, n = len(first), len(second)
        # if threshold = None, then double value is taken as a threshold
        # transduction, displaying characters at the same positions in each other
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
        # initialization of returned arrays
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
        Fills the _reversed_operation_costs array
        based on the available operation_costs array
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
        Calculates the maximum length of an element. Low
        in elementary transduction (up, low) for each up
        and the maximum length of the up element
        in elementary transduction (up, low) for each low
        """
        self.max_up_length = max(len(up) for up in self.operation_costs) if len(self.operation_costs) > 0 else -1
        self.max_low_length = (
            max(len(low) for low in self._reversed_operation_costs) if len(self._reversed_operation_costs) > 0 else -1
        )
        self.max_low_lengths_by_up, self.max_up_lengths_by_low = dict(), dict()
        for up, costs in self.operation_costs.items():
            self.max_low_lengths_by_up[up] = max(len(low) for low in costs) if len(costs) > 0 else -1
        for low, costs in self._reversed_operation_costs.items():
            self.max_up_lengths_by_low[low] = max(len(up) for up in costs) if len(costs) > 0 else -1

    def _backtraces_to_transductions(self, first, second, backtraces, threshold, return_cost=False):
        """
        Recovers transductions using the backlink table

        Arguments:
        ----------
        first, second: string
            upper and lower transduction elements
        backtraces: array-like, dtype = list, shape = (len (first) +1, len (second) +1)
            backlink table
        threshold: float
            threshold for screening transductions,
            only transductions worth <= threshold are returned
        return_cost: bool (optional, default = False)
            if True, then along with transductions, their value is returned

        Returns:
        -----------
        result: list
            a list of the form [(transduction, cost)] if return_cost = True
            and type [transduction] if return_cost = False,
            containing all transductions converting first to second,
            whose value does not exceed threshold
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

    def _perform_insertions(self, initial, max_cost):
        """
        returns all transductions values <= max_cost,
        which can be obtained from the elements of `initial`

        Arguments:
        ----------
        initial: list of tuples
            list of source transductions of the form [(transduction, cost)]
        max_cost: float
            maximum cost of transduction

        Returns:
        -----------
        final: list of tuples
            final list of transductions of the form [(transduction, cost)]
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
