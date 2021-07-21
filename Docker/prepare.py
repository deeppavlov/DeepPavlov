from main import initial_setup
from deeppavlov import deep_download

if __name__ == '__main__':
    deep_download('entity_linking_vx_siam.json')
    initial_setup()
