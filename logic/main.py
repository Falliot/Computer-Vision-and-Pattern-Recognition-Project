import math
import cv2
import cv2.aruco as aruco
import numpy as np
import os
from openpyxl import load_workbook
import matplotlib.pyplot as plt

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


def augmentAruco(markerCorner, id, img) :
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    # convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    # draw the bounding box of the ArUCo detection
    cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
    cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
    cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
    cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

    # draw the ArUco marker ID on the image
    cv2.putText(img, str(id), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    return img

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
            return (i['X'], i['Y'], i['Z'])

def main():
    # Change path to the video on your machine
    cap = cv2.VideoCapture('/Users/falli_ot/Desktop/P3-HJ1 (ArUco)/project.avi') # Due to an unknown reason the file was converted to .avi format
    augDict = loadAugImages('../CVAPR_Project/markers')
    listOf3DCoordinates = read3DAcuroCoordinates()
    listOfExisting3DCoordinates = []
    listOfRightCorners = []
    listOfIds = []

    cameraIntrinsics = np.array(
        [[1854, 0, 1920],
         [0, 1854, 1080],
         [0, 0, 1]], dtype=np.float32)
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

                   frame = augmentAruco(markerCorner, markerId, frame)

            cv2.imshow("Drone recording", frame)

            # End of frame
            for markerId in listOfIds:
                listOfExisting3DCoordinates.append(getExistingMarkers(markerId, listOf3DCoordinates))

            points2D = np.array(listOfRightCorners, dtype=np.float32)
            points3D = np.array(listOfExisting3DCoordinates, dtype=np.float32)

            # print("Count: ", len(listOfIds))
            # print("Count: ", len(points2D))
            # print("Count: ", len(points3D))

            if len(points3D) < 4:
                print("Not enought data to calculate the position")
            else:
                print("Calculating...")
                success, rotation_vector, translation_vector = cv2.solvePnP(points3D, points2D, cameraIntrinsics, None)

                rotM = cv2.Rodrigues(rotation_vector)[0]
                # print("Position: ", Rt)
                cameraPosition = -np.matrix(rotM).T * np.matrix(translation_vector)

                print("Position: ", cameraPosition)

                nose_end_point2D, jacobian = cv2.projectPoints(points3D, rotation_vector, translation_vector, cameraIntrinsics, None)
                # np.array([(0.0, 0.0, 1000.0)])

                # print("Results: ", nose_end_point2D)
                # print("Results: ", jacobian)

            if markerId in listOfIds:
                print("************************")
                # print(listOfIds)
                # print(points2D)
                # print(points3D)
                print("************************")

                listOfIds.clear()
                listOfRightCorners.clear()
                listOfExisting3DCoordinates.clear()

                print("--------------------------------")
                print("Empty ListOfIds: ", listOfIds)
                print("Empty ListOfCorners: ", listOfRightCorners)
                print("Empty ListOfExistingMarkers: ", listOfExisting3DCoordinates)

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

# Calculate the position and camera orientation
# def detectDronePosition(points3D, points_2D, listOfIds, cameraIntrinsicMatrix):



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
