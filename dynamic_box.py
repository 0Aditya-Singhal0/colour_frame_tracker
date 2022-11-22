import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)

def change_resolution(width,height):
    webcam.set(3,width)
    webcam.set(4,height)
    
# Start a while loop

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
    while True:
        _, imageFrame = webcam.read()
            
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
            
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                        (w, h),
                                        (0, 0, 255), 2)
                
            cv2.putText(imageFrame, "Red Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))

        
        # Program Termination
        cv2.imshow("Detection Window", imageFrame)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
    
if __name__ == '__main__':
    main()