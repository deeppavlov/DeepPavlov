from deeppavlov import build_model, train_model, train_evaluate_model_from_config
from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_readers.hallucination_detection_reader import HallucinationDatasetReader, RAGTruthDatasetReader


rs = []
for path in [
    "deeppavlov/configs/hallucination_detection/ragtruth_modernbert_large.json",
    "deeppavlov/configs/hallucination_detection/ragtruth_modernbert_base.json",
    "deeppavlov/configs/hallucination_detection/ragtruth_deberta_small.json",
]:
    config = read_json(path)
    
    results = train_evaluate_model_from_config(
        config,
        to_train=True,
        evaluation_targets=["valid", "test"],
        download=False,
        install=False
    )
    rs.append((path, results))
        
for path, results in rs:
    print("\n=== Результаты тренировки ===")
    for target, metrics in results.items():
        print(f"{target}: {metrics}")