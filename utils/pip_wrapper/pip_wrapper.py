import re
import subprocess
import sys


def install(*packages):
    return subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                  *[re.sub(r'\s', '', package) for package in packages]])
