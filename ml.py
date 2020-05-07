def accuracy(original, predicted):
    TRUE = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            TRUE += 1 if original[y][x] == predicted[y][x] else 0

    return TRUE / (original.shape[0] * original.shape[1])

def sensitivity(original, predicted):
    TP = 0
    FN = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            if original[y][x] == predicted[y][x] == 1:
                TP += 1
            elif original[y][x] == 1 and predicted[y][x] == 0:
                FN += 1

    return TP / (TP + FN)

def specificity(original, predicted):
    TN = 0
    FP = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            if original[y][x] == predicted[y][x] == 0:
                TN += 1
            elif original[y][x] == 0 and predicted[y][x] == 1:
                FP += 1

    return TN / (TN + FP)
