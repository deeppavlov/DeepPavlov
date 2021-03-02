from main import download_wikidata, parse_wikidata, State


if __name__ == '__main__':
    state = State.from_yaml()
    download_wikidata(state)
    parse_wikidata(state)
