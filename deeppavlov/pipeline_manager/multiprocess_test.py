from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import concurrent


def range_gen(n: int):
    for i in range(n):
        yield i + 1


def f(x):
    return x**2


if __name__ == '__main__':

    with ProcessPoolExecutor(3) as executor:
        futures = [executor.submit(f, x) for x in range_gen(5)]
        concurrent.futures.wait(futures)

        results = []
        for future in futures:
            result = future.result()
            results.append(result)

    print(results)

    # p = Pool(5)
    # print(p.map(f, range_gen(5)))


# from multiprocessing import Process
# import os
#
#
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     if hasattr(os, 'getppid'):  # only available on Unix
#         print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#
#
# def f(name):
#     info('function f')
#     print('hello', name)
#
#
# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()
