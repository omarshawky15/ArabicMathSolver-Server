import cv2
import numpy as np


def resize(image, dim=(32, 32)):
    bigger_dim = np.argmax(image.shape)
    pad_value = max(image.shape) - min(image.shape)
    if bigger_dim == 1:
        image = cv2.copyMakeBorder(image, pad_value // 2, pad_value // 2, 0, 0, cv2.BORDER_CONSTANT, value=255)
    else:
        image = cv2.copyMakeBorder(image, 0, 0, pad_value // 2, pad_value // 2, cv2.BORDER_CONSTANT, value=255)

    new_im = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return new_im


def preprocess_image(image, dim=(32, 32)):
    if image.shape != dim:
        image = resize(image, dim)
    return image


def crop_contour(out, mask):
    (y, x) = np.where(mask == 0)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    return out


def crop_image(path):
    im = cv2.imread(path)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = ~img
    ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort by x2 = x + w = boundingRect[0] + boundingRect[2]
    ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[2], reverse=True)
    crop_results = []
    rect_results = []
    for ctr in ctrs:
        mask = np.ones_like(img)
        cv2.drawContours(mask, ctr, -1, 0, thickness=10, lineType=cv2.LINE_AA)
        out = np.full(img.shape, 255)
        out[mask == 0] = ~img[mask == 0]
        out = crop_contour(out, mask)
        out = preprocess_image(out.astype('float32'))
        x1, y1, w, h = cv2.boundingRect(ctr)
        x2 = w + x1
        y2 = h + y1
        rect_results.append((x1, y1, x2, y2))
        crop_results.append(out)
    return np.array(crop_results), np.array(rect_results)
