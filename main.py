import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugImages(path):
    markersList = os.listdir(path)
    noOfMarkers =  len([name for name in markersList if name != ".DS_Store"])
    augDict = {}

    for imgPath in markersList:
        if imgPath != ".DS_Store":
            key = os.path.splitext(imgPath)[0]
            imgAug = cv2.imread(f'{path}/{imgPath}')
            augDict[key] = imgAug

    return augDict

# looks like from the video we dont check blue wall
def checkMarker(id):
    match id:
        case 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 76 | 77:
            return "red"
        case 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42:
            return "green"
        case 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 78 | 79 | 80 | 81 | 82 | 83:
            return "blue"
        case 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58| 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 | 75:
            return "orange"

def findArucoMarkers(frame, markerSize=4, totalMarkers=100, draw=True):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParams = aruco.DetectorParameters_create()
    boundingBox, ids, rejectedMarkers = aruco.detectMarkers(
        frameGray,
        arucoDict,
        parameters=arucoParams
    )
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(frame, boundingBox)

        return [boundingBox, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = int(bbox[0][1][0]), int(bbox[0][1][1])
    br = int(bbox[0][2][0]), int(bbox[0][2][1])
    bl = int(bbox[0][3][0]), int(bbox[0][3][1])

    h, w, c = imgAug.shape

    points1 = np.array([tl, tr, br, bl])
    points2 = np.float32([[0, 0],[w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(points2, points1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, points1.astype(int), (0, 0, 0)) # instead of green image it's possible to use also a green color, just change 0, 0, 0 to 0, 255, 0
    imgOut = img + imgOut


    if drawId:
        cv2.putText(
            imgOut,
            str(id),
            tl,
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 255, 255),
            2
        )

    return imgOut

def main():
    # Change path to the video on your machine
    cap = cv2.VideoCapture('/Users/falli_ot/Desktop/P3-HJ1 (ArUco)/project.avi') # Due to an unknown reason the file was converted to .avi format
    augDict = loadAugImages('markers')
    # Check if the video capture is open
    if(cap.isOpened() == False):
        print("Error opening video file")

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            arucoFound = findArucoMarkers(frame)

            # Loop through all the markers and augment each one
            if len(arucoFound[0]) != 0:
                for bbox, id in zip(arucoFound[0], arucoFound[1]):
                   color = checkMarker(id)
                   frame = augmentAruco(bbox, id, frame, augDict[color])
            cv2.imshow("Drone recording", frame)

            if cv2.waitKey(25) == ord('q'):
                print("exit")
                break
        else:
            print("break")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()