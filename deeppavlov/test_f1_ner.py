from deeppavlov.metrics.fmeasure import precision_recall_f1


test_samples = [
    {
        # true chunks: (6, 6), (7, 7); found chunks: (0, 6), (7, 7), (8, 8) 
        'y_true': ['O-TAG', 'O-TAG', 'O-TAG', 'O-TAG', 'O-TAG', 'O-TAG', 'E-TAG', 'E-TAG', 'O-TAG'],
        # true chunks: (3, 3), (6, 6), (7, 7); found chunks: (0, 6), (7, 7), (8, 8) 
        'y_pred': ['O-TAG', 'O-TAG', 'O-TAG', 'T-TAG', 'O-TAG', 'O-TAG', 'E-TAG', 'E-TAG', 'O-TAG'] , 
        'correct_metrics': {
            'precision': 66.66, 'recall': 100, 'f1': 80, 'tp': 2, 'tn': 0, 'fp': 1, 'fn': 0}
    }
]


for test_sample in test_samples:
    print("y_true:", test_sample['y_true'])
    print("y_pred:", test_sample['y_pred'])
    print(
        "calculated_metrics:\n", 
        precision_recall_f1(
            test_sample['y_true'], 
            test_sample['y_pred'], 
            print_results=False)
    )
    print(
        "correct metrics:\n",
        test_sample['correct_metrics']
    )
    print("#" * 40)
