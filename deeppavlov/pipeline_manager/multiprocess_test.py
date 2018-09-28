from multiprocessing import Pool


def range_gen(n: int):
    for i in range(n):
        yield (i + 1, 'text_{}'.format(i+1))


def f(x):
    return x[0]*x[0], x[1]


if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, range_gen(5)))


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
