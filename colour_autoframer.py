import cv2
# import mediapipe as mp
from vidstab import VidStab
from pyvirtualcam import PixelFormat
import pyvirtualcam
import platform
import numpy as np

# global variables
gb_zoom = 1.4

def zoom_at(image, coord=None, zoom_type=None):
    """
    Args:
        image: frame captured by camera
        coord: coordinates of face(nose)
        zoom_type:Is it a transition or normal zoom
    Returns:
        Image with cropped image
    """
    global gb_zoom
    # If zoom_type is transition check if Zoom is already done else zoom by 0.1 in current frame
    if zoom_type == 'transition' and gb_zoom < 3.0:
        gb_zoom = gb_zoom + 0.1

    # If zoom_type is normal zoom check if zoom more than 1.4 if soo zoom out by 0.1 in each frame
    if gb_zoom != 1.4 and zoom_type is None:
        gb_zoom = gb_zoom - 0.1

    zoom = gb_zoom
    # If coordinates to zoom around are not specified, default to center of the frame
    cy, cx = [i / 2 for i in image.shape[:-1]] if coord is None else coord[::-1]

    # Scaling the image using getRotationMatrix2D to appropriate zoom
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)

    # Use warpAffine to make sure that  lines remain parallel
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def get_red_contour(hsvFrame):
    
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
 
    kernal = np.ones((5, 5), "uint8")
    
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    # res_red = cv2.bitwise_and(imageFrame, imageFrame,mask = red_mask)
    
    
    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    return contours

def frame_manipulate(img):
    """
    Args:
        image: frame captured by camera
    Returns:
        Image with manipulated output
    """
    # # Mediapipe face set up
    # mp_face_detection = mp.solutions.face_detection
    # with mp_face_detection.FaceDetection(
    #         model_selection=1, min_detection_confidence=0.5) as face_detection:

    #     img.flags.writeable = False
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # results = face_detection.process(img)
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        
    contours = get_red_contour(hsvFrame)
    coords=[]

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            coords.append([x,y,x+w,y+h])
        if len(coords)==0:
            continue
        elif len(coords)==1:
            coords = np.array(coords)
            imageFrame = cv2.rectangle(imageFrame, (coords[0,0], coords[0,1]),
                                        (coords[0,2], coords[0,3]),
                                        (0, 0, 255), 2)
                
            cv2.putText(imageFrame, "Red Colour", (coords[0,0], coords[0,1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))	
        else:
            coords = np.array(coords)
            x = np.min(coords[:,0])
            y = np.min(coords[:,1])
            w = np.max(coords[:,2])
            h = np.max(coords[:,3])
    coordinates = None
    zoom_transition = None       

        # Perform zoom on the image
    img = zoom_at(img, coord=coordinates, zoom_type=zoom_transition)

    return img

def get_red_contour(hsvFrame):
    
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
 
    kernal = np.ones((5, 5), "uint8")
    
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    # res_red = cv2.bitwise_and(imageFrame, imageFrame,mask = red_mask)
    
    
    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    # Video Stabilizer
    device_val = None
    stabilizer = VidStab()

    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensions to cam object (not cap)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)

    # Check OS
    os = platform.system()
    if os == "Linux":
        device_val = "/dev/video2"

    # Start virtual camera
    with pyvirtualcam.Camera(1280, 720, 120, device=device_val, fmt=PixelFormat.BGR) as cam:
        print('Virtual camera device: ' + cam.device)

        while True:
            success, img = cap.read()
            img = frame_manipulate(img)
            # Stabilize the image to make sure that the changes with Zoom are very smooth
            img = stabilizer.stabilize_frame(input_frame=img,
                                             smoothing_window=2, border_size=-20)
            # Resize the image to make sure it does not crash pyvirtualcam
            img = cv2.resize(img, (1280, 720),
                             interpolation=cv2.INTER_CUBIC)

            cam.send(img)
            cam.sleep_until_next_frame()

if __name__ == "__main__":
    main()