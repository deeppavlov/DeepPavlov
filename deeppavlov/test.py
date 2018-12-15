from deeppavlov import build_model, configs
model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=True)
