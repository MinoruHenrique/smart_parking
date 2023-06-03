import cv2
import numpy as np
import argparse


parkings=[]
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        cv2.circle(image, [x,y], 0, (0,0,200), 6)
        refPt.append([x, y])
        cropping = False
        if len(refPt)%4 == 0:
            pt_A, pt_B, pt_C, pt_D = refPt
            refPt = np.array(refPt).reshape((-1,1,2))
            # print(refPt)
            cv2.polylines(image, [refPt], True, (0,0,255), 2)
            cv2.imshow("image", image)
            parkings.append(refPt)
            refPt = []
            #Warp perspective to show ROI
            print(pt_A, pt_B, pt_C, pt_D)
            # Here, I have used L2 norm. You can use L1 also.
            width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
            width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
            maxWidth = max(int(width_AD), int(width_BC))


            height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
            height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
            maxHeight = max(int(height_AB), int(height_CD))
            # maxWidth = 32
            # maxHeight = 54
            input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
            output_pts = np.float32([[0, 0],
                        [maxWidth-1, 0],
                        [maxWidth-1, maxHeight-1],
                        [0, maxHeight-1]])
            # Compute the perspective transform M
            M = cv2.getPerspectiveTransform(input_pts,output_pts)
            out = cv2.warpPerspective(clone,M,(maxWidth,maxHeight))
            cv2.imshow("ROI", out)
            cv2.waitKey(0)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-f", "--file", required=True, help="Path to spaces txt")
args = vars(ap.parse_args())
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        refPt=[]
        image = clone.copy()
	# if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
    elif key == ord("s"):
        with open(args["file"], 'w') as f:
            for park in parkings:
                A, B, C, D = park
                f.write(f"{A[0][0]} {A[0][1]} {B[0][0]} {B[0][1]} {C[0][0]} {C[0][1]} {D[0][0]} {D[0][1]}\n")
            
# if there are two reference points, then crop the region of interest
# from teh image and display it
# if len(refPt) == 2:
# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 	cv2.imshow("ROI", roi)
# 	cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()