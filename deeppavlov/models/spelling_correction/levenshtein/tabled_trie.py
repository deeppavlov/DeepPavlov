# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections import defaultdict

import numpy as np


class Trie:
    """
    Реализация префиксного бора (точнее, корневого направленного ациклического графа)

    Атрибуты
    --------
    alphabet: list, алфавит
    alphabet_codes: dict, словарь символ:код
    compressed: bool, индикатор сжатия
    cashed: bool, индикатор кэширования запросов к функции descend
    root: int, индекс корня
    graph: array, type=int, shape=(число вершин, размер алфавита), матрица потомков
    graph[i][j] = k <-> вершина k --- потомок вершины i по ребру, помеченному символом alphabet[j]
    data: array, type=object, shape=(число вершин), массив с данными, хранящямися в вершинах
    final: array, type=bool, shape=(число вершин), массив индикаторов
    final[i] = True <-> i --- финальная вершина
    """
    NO_NODE = -1
    SPACE_CODE = -1

    ATTRS = ['is_numpied', 'precompute_symbols', 'allow_spaces',
             'is_terminated', 'to_make_cashed']

    def __init__(self, alphabet, make_sorted=True, make_alphabet_codes=True,
                 is_numpied=False, to_make_cashed=False,
                 precompute_symbols=None, allow_spaces=False, dict_storage=False):
        self.alphabet = sorted(alphabet) if make_sorted else alphabet
        self.alphabet_codes = ({a: i for i, a in enumerate(self.alphabet)}
                               if make_alphabet_codes else self.alphabet)
        self.alphabet_codes[" "] = Trie.SPACE_CODE
        self.is_numpied = is_numpied
        self.to_make_cashed = to_make_cashed
        self.dict_storage = dict_storage
        self.precompute_symbols = precompute_symbols
        self.allow_spaces = allow_spaces
        self.initialize()

    def initialize(self):
        self.root = 0
        self.graph = [self._make_default_node()]
        self.data, self.final = [None], [False]
        self.nodes_number = 1
        self.descend = self._descend_simple
        self.is_terminated = False

    def _make_default_node(self):
        if self.dict_storage:
            return defaultdict(lambda: -1)
        elif self.is_numpied:
            return np.full(shape=(len(self.alphabet),),
                           fill_value=Trie.NO_NODE, dtype=int)
        else:
            return [Trie.NO_NODE] * len(self.alphabet)

    def save(self, outfile):
        """
        Сохраняет дерево для дальнейшего использования
        """
        with open(outfile, "w", encoding="utf8") as fout:
            attr_values = [getattr(self, attr) for attr in Trie.ATTRS]
            attr_values.append(any(x is not None for x in self.data))
            fout.write("{}\n{}\t{}\n".format(
                " ".join("T" if x else "F" for x in attr_values),
                self.nodes_number, self.root))
            fout.write(" ".join(str(a) for a in self.alphabet) + "\n")
            for index, label in enumerate(self.final):
                letters = self._get_letters(index, return_indexes=True)
                children = self._get_children(index)
                fout.write("{}\t{}\n".format(
                    "T" if label else "F", " ".join("{}:{}".format(*elem)
                                                    for elem in zip(letters, children))))
            if self.precompute_symbols is not None:
                for elem in self.data:
                    fout.write(":".join(",".join(
                        map(str, symbols)) for symbols in elem) + "\n")
        return

    def make_cashed(self):
        """
        Включает кэширование запросов к descend
        """
        self._descendance_cash = [dict() for _ in self.graph]
        self.descend = self._descend_cashed

    def make_numpied(self):
        self.graph = np.array(self.graph)
        self.final = np.asarray(self.final, dtype=bool)
        self.is_numpied = True

    def add(self, s):
        """
        Добавление строки s в префиксный бор
        """
        if self.is_terminated:
            raise TypeError("Impossible to add string to fitted trie")
        if s == "":
            self._set_final(self.root)
            return
        curr = self.root
        for i, a in enumerate(s):
            code = self.alphabet_codes[a]
            next = self.graph[curr][code]
            if next == Trie.NO_NODE:
                curr = self._add_descendant(curr, s[i:])
                break
            else:
                curr = next
        self._set_final(curr)
        return self

    def fit(self, words):
        for s in words:
            self.add(s)
        self.terminate()

    def terminate(self):
        if self.is_numpied:
            self.make_numpied()
        self.terminated = True
        if self.precompute_symbols is not None:
            precompute_future_symbols(self, self.precompute_symbols,
                                      allow_spaces=self.allow_spaces)
        if self.to_make_cashed:
            self.make_cashed()

    def __contains__(self, s):
        if any(a not in self.alphabet for a in s):
            return False
        # word = tuple(self.alphabet_codes[a] for a in s)
        node = self.descend(self.root, s)
        return (node != Trie.NO_NODE) and self.is_final(node)

    def words(self):
        """
        Возвращает итератор по словам, содержащимся в боре
        """
        branch, word, indexes = [self.root], [], [0]
        letters_with_children = [self._get_children_and_letters(self.root)]
        while len(branch) > 0:
            if self.is_final(branch[-1]):
                yield "".join(word)
            while indexes[-1] == len(letters_with_children[-1]):
                indexes.pop()
                letters_with_children.pop()
                branch.pop()
                if len(indexes) == 0:
                    raise StopIteration()
                word.pop()
            next_letter, next_child = letters_with_children[-1][indexes[-1]]
            indexes[-1] += 1
            indexes.append(0)
            word.append(next_letter)
            branch.append(next_child)
            letters_with_children.append(self._get_children_and_letters(branch[-1]))

    def is_final(self, index):
        """
        Аргументы
        ---------
        index: int, номер вершины

        Возвращает
        ----------
        True: если index --- номер финальной вершины
        """
        return self.final[index]

    def find_partitions(self, s, max_count=1):
        """
        Находит все разбиения s = s_1 ... s_m на словарные слова s_1, ..., s_m
        для m <= max_count
        """
        curr_agenda = [(self.root, [], 0)]
        for i, a in enumerate(s):
            next_agenda = []
            for curr, borders, cost in curr_agenda:
                if cost >= max_count:
                    continue
                child = self.graph[curr][self.alphabet_codes[a]]
                # child = self.graph[curr][a]
                if child == Trie.NO_NODE:
                    continue
                next_agenda.append((child, borders, cost))
                if self.is_final(child):
                    next_agenda.append((self.root, borders + [i + 1], cost + 1))
            curr_agenda = next_agenda
        answer = []
        for curr, borders, cost in curr_agenda:
            if curr == self.root:
                borders = [0] + borders
                answer.append([s[left:borders[i + 1]] for i, left in enumerate(borders[:-1])])
        return answer

    def __len__(self):
        return self.nodes_number

    def __repr__(self):
        answer = ""
        for i, (final, data) in enumerate(zip(self.final, self.data)):
            letters, children = self._get_letters(i), self._get_children(i)
            answer += "{0}".format(i)
            if final:
                answer += "F"
            for a, index in zip(letters, children):
                answer += " {0}:{1}".format(a, index)
            answer += "\n"
            if data is not None:
                answer += "data:{0} {1}\n".format(len(data), " ".join(str(elem) for elem in data))
        return answer

    def _add_descendant(self, parent, s, final=False):
        for a in s:
            code = self.alphabet_codes[a]
            parent = self._add_empty_child(parent, code, final)
        return parent

    def _add_empty_child(self, parent, code, final=False):
        """
        Добавление ребёнка к вершине parent по символу с кодом code
        """
        self.graph[parent][code] = self.nodes_number
        self.graph.append(self._make_default_node())
        self.data.append(None)
        self.final.append(final)
        self.nodes_number += 1
        return (self.nodes_number - 1)

    def _descend_simple(self, curr, s):
        """
        Спуск из вершины curr по строке s
        """
        for a in s:
            curr = self.graph[curr][self.alphabet_codes[a]]
            if curr == Trie.NO_NODE:
                break
        return curr

    def _descend_cashed(self, curr, s):
        """
        Спуск из вершины curr по строке s с кэшированием
        """
        if s == "":
            return curr
        curr_cash = self._descendance_cash[curr]
        answer = curr_cash.get(s, None)
        if answer is not None:
            return answer
        # для оптимизации дублируем код
        res = curr
        for a in s:
            res = self.graph[res][self.alphabet_codes[a]]
            # res = self.graph[res][a]
            if res == Trie.NO_NODE:
                break
        curr_cash[s] = res
        return res

    def _set_final(self, curr):
        """
        Делает состояние curr завершающим
        """
        self.final[curr] = True

    def _get_letters(self, index, return_indexes=False):
        """
        Извлекает все метки выходных рёбер вершины с номером index
        """
        if self.dict_storage:
            answer = list(self.graph[index].keys())
        else:
            answer = [i for i, elem in enumerate(self.graph[index])
                      if elem != Trie.NO_NODE]
        if not return_indexes:
            answer = [(self.alphabet[i] if i >= 0 else " ") for i in answer]
        return answer

    def _get_children_and_letters(self, index, return_indexes=False):
        if self.dict_storage:
            answer = list(self.graph[index].items())
        else:
            answer = [elem for elem in enumerate(self.graph[index])
                      if elem[1] != Trie.NO_NODE]
        if not return_indexes:
            for i, (letter_index, child) in enumerate(answer):
                answer[i] = (self.alphabet[letter_index], child)
        return answer

    def _get_children(self, index):
        """
        Извлекает всех потомков вершины с номером index
        """
        if self.dict_storage:
            return list(self.graph[index].values())
        else:
            return [elem for elem in self.graph[index] if elem != Trie.NO_NODE]


class TrieMinimizer:
    def __init__(self):
        pass

    def minimize(self, trie, dict_storage=False, make_cashed=False, make_numpied=False,
                 precompute_symbols=None, allow_spaces=False, return_groups=False):
        N = len(trie)
        if N == 0:
            raise ValueError("Trie should be non-empty")
        node_classes = np.full(shape=(N,), fill_value=-1, dtype=int)
        order = self.generate_postorder(trie)
        # processing the first node
        index = order[0]
        node_classes[index] = 0
        class_representatives = [index]
        node_key = ((), (), trie.is_final(index))
        classes, class_keys = {node_key: 0}, [node_key]
        curr_index = 1
        for index in order[1:]:
            letter_indexes = tuple(trie._get_letters(index, return_indexes=True))
            children = trie._get_children(index)
            children_classes = tuple(node_classes[i] for i in children)
            key = (letter_indexes, children_classes, trie.is_final(index))
            key_class = classes.get(key, None)
            if key_class is not None:
                node_classes[index] = key_class
            else:
                # появился новый класс
                class_keys.append(key)
                classes[key] = node_classes[index] = curr_index
                class_representatives.append(curr_index)
                curr_index += 1
        # построение нового дерева
        compressed = Trie(trie.alphabet, is_numpied=make_numpied,
                          dict_storage=dict_storage, allow_spaces=allow_spaces,
                          precompute_symbols=precompute_symbols)
        L = len(classes)
        new_final = [elem[2] for elem in class_keys[::-1]]
        if dict_storage:
            new_graph = [defaultdict(int) for _ in range(L)]
        elif make_numpied:
            new_graph = np.full(shape=(L, len(trie.alphabet)),
                                fill_value=Trie.NO_NODE, dtype=int)
            new_final = np.array(new_final, dtype=bool)
        else:
            new_graph = [[Trie.NO_NODE for a in trie.alphabet] for i in range(L)]
        for (indexes, children, final), class_index in \
                sorted(classes.items(), key=(lambda x: x[1])):
            row = new_graph[L - class_index - 1]
            for i, child_index in zip(indexes, children):
                row[i] = L - child_index - 1
        compressed.graph = new_graph
        compressed.root = L - node_classes[trie.root] - 1
        compressed.final = new_final
        compressed.nodes_number = L
        compressed.data = [None] * L
        if make_cashed:
            compressed.make_cashed()
        if precompute_symbols is not None:
            if (trie.is_terminated and trie.precompute_symbols
                    and trie.allow_spaces == allow_spaces):
                # копируем будущие символы из исходного дерева
                # нужно, чтобы возврат из финальных состояний в начальное был одинаковым в обоих деревьях
                for i, node_index in enumerate(class_representatives[::-1]):
                    # будущие символы для представителя i-го класса
                    compressed.data[i] = copy.copy(trie.data[node_index])
            else:
                precompute_future_symbols(compressed, precompute_symbols, allow_spaces)
        if return_groups:
            node_classes = [L - i - 1 for i in node_classes]
            return compressed, node_classes
        else:
            return compressed

    def generate_postorder(self, trie):
        """
        Обратная топологическая сортировка
        """
        order, stack = [], []
        stack.append(trie.root)
        colors = ['white'] * len(trie)
        while len(stack) > 0:
            index = stack[-1]
            color = colors[index]
            if color == 'white':  # вершина ещё не обрабатывалась
                colors[index] = 'grey'
                for child in trie._get_children(index):
                    # проверяем, посещали ли мы ребёнка раньше
                    if child != Trie.NO_NODE and colors[child] == 'white':
                        stack.append(child)
            else:
                if color == 'grey':
                    colors[index] = 'black'
                    order.append(index)
                stack = stack[:-1]
        return order


def load_trie(infile):
    with open(infile, "r", encoding="utf8") as fin:
        line = fin.readline().strip()
        flags = [x == 'T' for x in line.split()]
        if len(flags) != len(Trie.ATTRS) + 1:
            raise ValueError("Wrong file format")
        nodes_number, root = map(int, fin.readline().strip().split())
        alphabet = fin.readline().strip().split()
        trie = Trie(alphabet)
        for i, attr in enumerate(Trie.ATTRS):
            setattr(trie, attr, flags[i])
        read_data = flags[-1]
        final = [False] * nodes_number
        # print(len(alphabet), nodes_number)
        if trie.dict_storage:
            graph = [defaultdict(lambda: -1) for _ in range(nodes_number)]
        elif trie.is_numpied:
            final = np.array(final)
            graph = np.full(shape=(nodes_number, len(alphabet)),
                            fill_value=Trie.NO_NODE, dtype=int)
        else:
            graph = [[Trie.NO_NODE for a in alphabet] for i in range(nodes_number)]
        for i in range(nodes_number):
            line = fin.readline().strip()
            if "\t" in line:
                label, transitions = line.split("\t")
                final[i] = (label == "T")
            else:
                label = line
                final[i] = (label == "T")
                continue
            transitions = [x.split(":") for x in transitions.split()]
            for code, value in transitions:
                graph[i][int(code)] = int(value)
        trie.graph = graph
        trie.root = root
        trie.final = final
        trie.nodes_number = nodes_number
        trie.data = [None] * nodes_number
        if read_data:
            for i in range(nodes_number):
                line = fin.readline().strip("\n")
                trie.data[i] = [set(elem.split(",")) for elem in line.split(":")]
        if trie.to_make_cashed:
            trie.make_cashed()
        return trie


def make_trie(alphabet, words, compressed=True, is_numpied=False,
              make_cashed=False, precompute_symbols=False,
              allow_spaces=False, dict_storage=False):
    trie = Trie(alphabet, is_numpied=is_numpied, to_make_cashed=make_cashed,
                precompute_symbols=precompute_symbols, dict_storage=dict_storage)
    trie.fit(words)
    if compressed:
        tm = TrieMinimizer()
        trie = tm.minimize(trie, dict_storage=dict_storage, make_cashed=make_cashed,
                           make_numpied=is_numpied, precompute_symbols=precompute_symbols,
                           allow_spaces=allow_spaces)
    return trie


def precompute_future_symbols(trie, n, allow_spaces=False):
    """
    Collecting possible continuations of length <= n for every node
    """
    if n == 0:
        return
    if trie.is_terminated and trie.precompute_symbols:
        # символы уже предпосчитаны
        return
    for index, final in enumerate(trie.final):
        trie.data[index] = [set() for i in range(n)]
    for index, (node_data, final) in enumerate(zip(trie.data, trie.final)):
        node_data[0] = set(trie._get_letters(index))
        if allow_spaces and final:
            node_data[0].add(" ")
    for d in range(1, n):
        for index, (node_data, final) in enumerate(zip(trie.data, trie.final)):
            children = set(trie._get_children(index))
            for child in children:
                node_data[d] |= trie.data[child][d - 1]
            # в случае, если разрешён возврат по пробелу в стартовое состояние
            if allow_spaces and final:
                node_data[d] |= trie.data[trie.root][d - 1]
    trie.terminated = True
