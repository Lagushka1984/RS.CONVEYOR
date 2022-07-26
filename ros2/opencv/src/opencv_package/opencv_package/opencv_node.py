from typing import Tuple
import cv2
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
import json


class OpenCV:
    bins: np.ndarray = [0, 51, 102, 153, 204, 255]
    blrs: int = 20
    sizeError: int = 10
    surfaceError: int = 100
    colorError: int = 7

    def __init__(self, path) -> None:
        self.objects = json.load(open(path))

    def bgRemove(self, img: np.ndarray) -> np.ndarray:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img[:, :, :] = np.digitize(img[:, :, :], self.bins, right=True) * 51
        myimage_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, background = cv2.threshold(myimage_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        return background

    def findRect(self, img: np.ndarray) -> Tuple[ndarray, tuple]:
        for _ in range(self.blrs):
            img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            w, h = rect[1]
            if h < 200 and w < 200:
                if h > 20 or w > 20:
                    return box, rect

    def collectObject(self, img: np.ndarray, debug=False, points=None) -> Tuple[ndarray, str]:
        bg = self.bgRemove(img)
        box, rect = self.findRect(bg)
        rotated = self.rotateAndCutImage(img, rect)
        width = int(rect[1][0])
        height = int(rect[1][1])
        surface = width * height
        color = self.averageColor(rotated)
        params = {'width': width,
                  'height': height,
                  'surface': surface,
                  'average_color': color}
        name = self.compareObjects(params)
        if debug:
            print(name, params)
        return self.convertBox(box, points=points), name

    def compareObjects(self, params: dict) -> str:
        rParams = int(params['average_color'][0])
        gParams = int(params['average_color'][1])
        bParams = int(params['average_color'][2])
        size = (params['width'], params['height'])
        width = min(size)
        height = max(size)
        surface = params['surface']
        for object in self.objects['objects']:
            error = abs(object['width'] - width)
            if error > self.sizeError:
                continue
            error = abs(object['height'] - height)
            if error > self.sizeError:
                continue
            # error = abs(object['surface'] - surface)
            # if error > self.surfaceError:
            #    continue
            rObject = eval(object['average_color'])[0]
            gObject = eval(object['average_color'])[1]
            bObject = eval(object['average_color'])[2]
            error = abs(rObject - rParams)
            if error > self.colorError:
                continue
            error = abs(gObject - gParams)
            if error > self.colorError:
                continue
            error = abs(bObject - bParams)
            if error > self.colorError:
                continue
            return object['name']
        return 'unknown'

    def rotateObject(self, img: np.ndarray) -> np.ndarray:
        return self.rotateAndCutImage(img, self.findRect(self.bgRemove(img))[1])

    def debug(self, img: np.ndarray, name='default') -> None:
        bg = self.bgRemove(img)
        box, rect = self.findRect(bg)
        width = int(rect[1][0])
        height = int(rect[1][1])
        surface = width * height
        color = self.averageColor(self.rotateObject(img))
        print()
        print(f'name : {name}')
        print(f'width : {width}')
        print(f'height : {height}')
        print(f'surface : {surface}')
        print(f'color : {color}')
        print()

    @staticmethod
    def convertBox(box: np.ndarray, points=None) -> np.ndarray:
        if points is None:
            points = [0, 0, 0, 0]
        newBox = box.astype(int)
        for i in range(len(newBox)):
            newBox[i][0] += points[2]
            newBox[i][1] += points[0]
        newBox = newBox.reshape((-1, 1, 2))
        return newBox

    @staticmethod
    def rotateAndCutImage(img: np.ndarray, rect: tuple) -> np.ndarray:
        width = int(rect[1][0])
        height = int(rect[1][1])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        return warped

    @staticmethod
    def averageColor(img: np.ndarray) -> np.ndarray:
        average_color_row = np.average(img, axis=0)
        average_color = np.average(average_color_row, axis=0)
        return average_color.astype(int)

    @staticmethod
    def dominantColors(img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(img)
        color = kmeans.cluster_centers_
        return color.astype(int)


def main():
    points = [190, 410, 250, 430]
    amount = 13
    path = 'objects.json'
    opencv = OpenCV(path)
    for i in range(amount):
        img = cv2.imread(f'{i}.jpg')
        buffer, name = opencv.collectObject(img[points[0]:points[1], points[2]:points[3]], debug=False,
                                            points=points)
        cv2.rectangle(img, (points[2], points[0]), (points[3], points[1]), (0, 0, 255), 2)
        img = cv2.polylines(img, [buffer], True, (255, 0, 0), 2)
        cv2.imshow(f'{name} / {i}.jpg', img)
    cv2.waitKey(500000)


if __name__ == '__main__':
    main()
