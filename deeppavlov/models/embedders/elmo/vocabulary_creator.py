from multiprocessing import Pool
import collections


def counters_merge(counters):
    if len(counters) < 3:
        share_counter = collections.Counter()
        for counter in counters:
            share_counter += counter
        return share_counter
    else:
        split_point = len(counters)//2
        l_c = counters_merge(counters[:split_point])
        r_c = counters_merge(counters[split_point:])
        return l_c + r_c


def count_items(lines):
    counters = []
    for line in lines:
        counters.append(collections.Counter(line))
    return [counters_merge(counters)]


def chunk_generator(item_list, chunk_size):
    for i in range(0, len(item_list), chunk_size):
        yield item_list[i:i + chunk_size]


# def lines_token_count(text_lines):
#     return count_items(lines)


def start_pool(items, func, cpu_n):
    with Pool(cpu_n) as p:
        results = []
        for result in p.imap_unordered(func, items):
            results.extend(result)
    return results


def create_vocab(txt_lines, cpu_n=1, min_line_per_worker=10000):
    items_per_process = max(len(txt_lines)//cpu_n, min_line_per_worker)
    chunk_gen = chunk_generator(txt_lines, items_per_process)
    token_counters = start_pool(list(chunk_gen), count_items, cpu_n)
    sum_token_counter = sum(token_counters, collections.Counter())
    vocab = ['<S>', '</S>', '<UNK>'] + [token for token, c in sum_token_counter.most_common()]
    return vocab
