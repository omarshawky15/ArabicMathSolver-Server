from parsing import Symbol_Info, all_labels


# Labels to Symbol_info and classify
def classify(model, images, most_probable_k=3):
    eqn_predictions = model.predict(images)
    most_probable = eqn_predictions.argsort(axis=1)[:, -most_probable_k:]
    return most_probable


def labels_to_symbols(rects, pred_labels):
    symbols = []
    length = len(pred_labels) - 1
    for i in range(length + 1):
        x1, y1, x2, y2 = rects[length - i]
        symbols.append(Symbol_Info(all_labels[pred_labels[length - i]], x1, y1, x2, y2))
    return symbols
