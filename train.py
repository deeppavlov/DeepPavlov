from deeppavlov import build_model, train_model, train_evaluate_model_from_config
from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_readers.hallucination_detection_reader import HallucinationDatasetReader, RAGTruthDatasetReader
config = read_json("deeppavlov/configs/hallucination_detection/ragtruth.json")
results = train_evaluate_model_from_config(
    config,
    to_train=True,
    # evaluation_targets=["valid", "train", "test"],
    download=False,
    install=False
)

print("\n=== Результаты тренировки ===")
for target, metrics in results.items():
    print(f"{target}: {metrics}")