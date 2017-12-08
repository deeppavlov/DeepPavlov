import re

def tokenize(s):
    return re.findall(r"[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)
