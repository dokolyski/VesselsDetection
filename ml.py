def accuracy(original, predicted, importanceMask = []):
    TRUE = 0
    size = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            multiplier = 1
            if importanceMask != []:
                multiplier = importanceMask[y][x]
            TRUE += 1 * multiplier if original[y][x] == predicted[y][x] else 0
            size += multiplier

    return TRUE / size

def sensitivity(original, predicted, importanceMask = []):
    TP = 0
    FN = 0
    size = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            if importanceMask != [] and importanceMask[y][x] == 1:
                if original[y][x] == predicted[y][x] == 1:
                    TP += 1
                elif original[y][x] == 1 and predicted[y][x] == 0:
                    FN += 1

    return TP / (TP + FN)

def specificity(original, predicted, importanceMask = []):
    TN = 0
    FP = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            if importanceMask != [] and importanceMask[y][x] == 1:
                if original[y][x] == predicted[y][x] == 0:
                    TN += 1
                elif original[y][x] == 0 and predicted[y][x] == 1:
                    FP += 1

    return TN / (TN + FP)
