import numpy as np
import os

def parse_result(path_csv):
    with open(path_csv) as f:
        probs = []
        max_indexes = []
        for line in f:
            line = line.strip().split(',')
            index = int(line[0])
            prob = np.array(line[1:], np.float32)
            max_index = np.argmax(prob)
            max_indexes.append(max_index)
            probs.append(prob)
        probs = np.array(probs, np.float32)
    
    print(os.path.basename(path_csv))
    return probs

def ensemble(paths):
    model_probs = []
    for path in paths:
        probs = parse_result(path)
        model_probs.append(probs)
    model_probs = np.array(model_probs)
    ensemble_probs = np.mean(model_probs, axis=0)
    ensemble_pred = np.argmax(ensemble_probs, axis=1)
    return ensemble_pred

if __name__ == '__main__':
    path_csv = 'train_df.csv'
    labels = []
    with open(path_csv) as f:
        f.__next__()
        for line in f:
            index, file_name, class_, state, label = line.strip().split(',')
            labels.append(label)

    ids = np.unique(labels)
    label_dict = list(ids)

    paths = [
    'convnext_test_scores.csv',
    'convnext_test_scores.csv',
    'convnext_test_scores.csv',
    'convnext_test_scores.csv',
    'convnext_test_scores.csv',
    'model_b5_new_aug_smoothing_hierachy_00.pth_result.csv',
    'model_b5_new_aug_smoothing_hierachy_01.pth_result.csv',
    'model_b5_new_aug_smoothing_hierachy_02.pth_result.csv',
    'model_b5_new_aug_smoothing_hierachy_03.pth_result.csv',
    'model_b5_new_aug_smoothing_hierachy_04.pth_result.csv',
    ]

    ensemble_preds = ensemble(paths)

    with open('submission.csv', 'w') as f:
        f.write('index,label\n')
        for i, pred in enumerate(ensemble_preds):
            f.write('%d,%s\n' % (i, label_dict[pred]))