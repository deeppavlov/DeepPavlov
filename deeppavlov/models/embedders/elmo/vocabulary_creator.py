"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from multiprocessing import Pool
import collections

def chunk_generator(items_list, chunk_size):
    for i in range(0, len(items_list), chunk_size):
        yield items_list[i:i + chunk_size]

def count_items(lines):
    counters = []
    for line in lines:
        counters.append(collections.Counter(line))
    while len(counters) > 1:
        count_pairs_gen = chunk_generator(counters, 2)
        counters = []
        for count_pair in count_pairs_gen:
            counters.append(sum(count_pair, collections.Counter()))
    return counters

def start_pool(items, func, worker_n):
    with Pool(worker_n) as p:
        results = []
        for result in p.imap_unordered(func, items):
            results.extend(result)
    return results

def create_vocab(txt_lines:list, worker_n=1, min_line_per_worker=100000):
    """
    txt_lines = list of list of tokens
    """
    items_per_process = max(len(txt_lines)//worker_n, min_line_per_worker)
    chunk_gen = chunk_generator(txt_lines, items_per_process)
    token_counters = start_pool(chunk_gen, count_items, worker_n)
    sum_token_counter = sum(token_counters, collections.Counter())
    vocab = ['<S>', '</S>', '<UNK>'] + [token for token, c in sum_token_counter.most_common()]
    return vocab
