import cv2
import time, datetime
import threading as T

def detect(frame, list_of):
    for (x, y, width, height) in list_of:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
        # frame : image
        # (x, y) , (x + width, y + height) : position of rectangle
        # (255, 0, 0) : color of rectangle
        # 3 : thickness of rectangle is 3 pixel    
        
def main():
    capture = cv2.VideoCapture(0) # Access to my camera I have one camera and index is 0.
    face_cc = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    body_cc = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
    eye_cc = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')    

    while True:
        _, frame = capture.read() # Return _ : somethings , frame : frames of video.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Get gray color to all frames. COLOR_BGR2GRAY : get gray color
        faces = face_cc.detectMultiScale(gray, 1.5, 5) # Return list of faces could be captured.
        # 1.3 : accurcy and speed of algorithm (lower is faster).
        # 5 : how many face can detect? 5 is middle (lower than 5 decrease power detect / more than 5 increase power detect)
        bodys = body_cc.detectMultiScale(gray, 1.5, 5) # Return list of bodys could be captured.
        eyes = eye_cc.detectMultiScale(gray, 1.5, 5) # Return list of eyes could be captured.

        # multi threading
        t1 = T.Thread(target=detect, args=[frame, faces])        
        t2 = T.Thread(target=detect, args=[frame, bodys])
        t3 = T.Thread(target=detect, args=[frame, eyes])
        t1.start(), t2.start(), t3.start()

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'): # If i press q on keyboard capture is going to end.
            break
    capture.release() # Release my Camera
    cv2.destroyAllWindows() # Kill

if __name__ == "__main__":
    main() # I Love C, Python 
