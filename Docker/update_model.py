from main import download_wikidata, parse_wikidata, State


if __name__ == '__main__':
    state = State.from_yaml()
    parse_entities(state)
    update_faiss(state)
