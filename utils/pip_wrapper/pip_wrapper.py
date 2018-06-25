import subprocess
import sys


def install(package):
    return subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
