import pip

with open("requirements.txt") as f:
    for line in f:
            pip.main(['install', '-U', line])
