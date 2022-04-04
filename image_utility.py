import cv2
import numpy as np


def resize(image, dim=(32, 32)):
    old_size = image.shape[:2]

    ratio = float(dim[0]) / max(old_size)
    new_size = tuple([max(int(x * ratio), 1) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = dim[0] - new_size[1]
    delta_h = dim[1] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [256, 256]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def preprocess_image(image, dim=(32, 32)):
    # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
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
    # img = cv2.Canny(img, 100, 200)
    # cv2.imshow(img)
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
        # cv2_imshow(out)
        x1, y1, w, h = cv2.boundingRect(ctr)
        x2 = w + x1
        y2 = h + y1
        rect_results.append((x1, y1, x2, y2))
        crop_results.append(out)
    return np.array(crop_results), np.array(rect_results)