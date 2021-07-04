from main import parse_entities, update_faiss, State


if __name__ == '__main__':
    state = State.from_yaml()
    parse_entities(state)
    update_faiss(state)
