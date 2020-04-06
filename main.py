import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import predictor


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img_noisy = img + noise * np.random.rand(*img.shape)

    return (row, col, rad), img_noisy, img


def find_circle(img):
    # Fill in this function
    detect = predictor.circle_find(img)
    return detect


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    for attempt in range(1):
        results = []
        for _ in tqdm(range(1000)):
            params, img = noisy_circle(200, 50, 2)
            detected = find_circle(img)
            results.append(iou(params, detected))
        results = np.array(results)
        # print((results > 0.7))
        print('Accuracy:', (results > 0.5).mean(), 'attempt ', attempt+1)
        y, x, r = params
        img_rec = cv2.rectangle(img_noisy.copy(), (x - r - 1, y - r - 1), (x + r + 1, y + r + 1), (0, 0, 0), 2)
        stacked_img = np.hstack([img, img_noisy, np.zeros([200, 1], dtype=np.uint8), img_rec])
        plt.imshow(stacked_img)
        plt.show()

main()
