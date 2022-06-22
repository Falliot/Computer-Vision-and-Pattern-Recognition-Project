import math
import cv2
import cv2.aruco as aruco
import numpy as np
import os
from openpyxl import load_workbook

from logic import utils


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


def augmentAruco(markerCorner, id, img, imgAug, drawId=True):
    tl = int(markerCorner[0][0][0]), int(markerCorner[0][0][1])
    tr = int(markerCorner[0][1][0]), int(markerCorner[0][1][1])
    br = int(markerCorner[0][2][0]), int(markerCorner[0][2][1])
    bl = int(markerCorner[0][3][0]), int(markerCorner[0][3][1])

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

def read3DAcuroCoordinates():
    # Load Excel file with Markers
    markersData = load_workbook('../CVAPR_Project/data/ArUco_markers_3D.xlsx')
    sheet = markersData.active
    rows = sheet.rows
    headers = [cell.value for cell in next(rows)]
    headers.pop()
    allRows = []
    for row in rows:
        data = {}
        for title, cell in zip(headers, row):  # ('item', cell_1)
            data[title] = cell.value
        allRows.append(data)
    return allRows

def getExistingMarkers(markerId, listOf3DCoordinates):
    for i in listOf3DCoordinates:
        if i['Marker_ID'] == markerId:
            return dict(i)

def main():
    # Change path to the video on your machine
    cap = cv2.VideoCapture('/Users/falli_ot/Desktop/P3-HJ1 (ArUco)/project.avi') # Due to an unknown reason the file was converted to .avi format
    augDict = loadAugImages('../CVAPR_Project/markers')
    listOf3DCoordinates = read3DAcuroCoordinates()
    listWithExistingMarkers = []
    listOfRightCorners = []
    listOfIds = []

    cameraIntrinsics = np.array([[1854, 0, 0], [0, 1854, 0], [1920, 1080, 1]])
    # np.set_printoptions(suppress=True)
    print(cameraIntrinsics)

    # Check if the video capture is open
    if(cap.isOpened() == False):
        print("Error opening video file")

    while(cap.isOpened()):
        # Read frame:
        ret, frame = cap.read()

        if ret == True:
            arucoFound = findArucoMarkers(frame)

            # Loop through all the markers and augment each one
            if len(arucoFound[0]) != 0:
                for markerCorner, markerId in zip(arucoFound[0], arucoFound[1]):
                  # print('id: ', markerId)
                   # extract the marker corners (which are always returned in
                   # top-left, top-right, bottom-right, and bottom-left order)
                   corners = markerCorner.reshape((4, 2))
                  # print("Corners", corners)
                   (topLeft, topRight, bottomRight, bottomLeft) = corners

                   # convert each of the (x, y)-coordinate pairs to integers
                   topRight = (int(topRight[0]), int(topRight[1]))
                   listOfRightCorners.append(topRight)
                   listOfIds.append(int(markerId))
                   # print("ListOfIds: ", listOfIds)
                   # print("ListOfCorners: ", listOfRightCorners)

                   color = checkMarker(markerId)
                   frame = augmentAruco(markerCorner, markerId, frame, augDict[color])

                   for markerId in listOfIds:
                       listWithExistingMarkers.append(getExistingMarkers(markerId, listOf3DCoordinates))

            cv2.imshow("Drone recording", frame)

            # End of frame
            if markerId in listOfIds:
                print("************************")
                print(listOfIds)
                print(listOfRightCorners)
                print(listWithExistingMarkers)
                print("************************")

                listOfIds.clear()
                listOfRightCorners.clear()
                listWithExistingMarkers.clear()

                print("--------------------------------")
                print("Empty ListOfIds: ", listOfIds)
                print("Empty ListOfCorners: ", listOfRightCorners)
                print("Empty ListOfExistingMarkers: ", listWithExistingMarkers)

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

# camera intrinsic matrix python
#  principal point:  divide by 2  -> cx = w/2
#fov*pi/180
#focal length = xf = cx/tan(fov_rad/2)

# corners - no storing, rewriting the data each frame (in  open cv it returns 4 corners, use ONLY right upper corner)
# check the number of detected corners, if it's < 4 dont perform calculations else
# calculate the position and camera orientation

# arguments: 3D coordinates, corners, ids, intrinsics,
#estimateWorldCameraPose(set of image points(tuple of 2D, 3D coordicated, intrisics)) returns world Location

# imagePoints = zeros(2, n) -> n nu,ber of arucon in the frame
# plot 3D




# [[1854    0    0]
#  [   0 1854    0]
#  [1920 1080    1]]


  # Camera intrinsic
  #   w = 3840
  #   h = 2160
  #   cx = w/2 # Principal point -> optical center
  #   cy = h/2
  #   fov = 92
  #   fovRad = fov * math.pi / 180
  #   fx = cx / math.tan(fovRad / 2) # focal length
  #   fy = cx / math.tan(fovRad / 2)
  #   intrinsic = get_camera_matrix(w, h, fov)
  # print(intrinsic)
