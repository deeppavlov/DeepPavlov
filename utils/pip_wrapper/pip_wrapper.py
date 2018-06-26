import subprocess
import sys


def install(package):
    return subprocess.check_call(f'{sys.executable} -m pip install {package}', shell=True)
