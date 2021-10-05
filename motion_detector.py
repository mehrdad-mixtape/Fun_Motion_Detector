import cv2
import time, datetime


def main():
    capture = cv2.VideoCapture(0) # Access to my camera I have one camera and index is 0.
    
    while True:
        _, frame = capture.read() # Return _ : somethings , frame : frames of video.
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'): # If i press q on keyboard capture is going to end.
            break
    capture.release() # Release my Camera
    cv2.destroyAllWindows() # Kill

if __name__ == "__main__":
    main() # I Love C, Python 
