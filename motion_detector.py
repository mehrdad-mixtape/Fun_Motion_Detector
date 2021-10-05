import cv2
import time, datetime

def main():
    capture = cv2.VideoCapture(0) # Access to my camera I have one camera and index is 0.
    face_cc = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    body_cc = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')

    recording = True
    frame_size = (int(capture.get(3)), int(capture.get(4)))
    frame_rate = 20.0
    video_format = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(f"video_{datetime.datetime.now()}.mp4", video_format, frame_rate, frame_size)

    while True:
        _, frame = capture.read() # Return _ : somethings , frame : frames of video.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Get gray color to all frames. COLOR_BGR2GRAY : get gray color
        faces = face_cc.detectMultiScale(gray, 1.3, 5) # Return list of faces could be captured.
        # 1.3 : accurcy and speed of algorithm (lower is faster).
        # 5 : how many face can detect? 5 is middle (lower than 5 decrease power detect / more than 5 increase power detect)
        bodys = body_cc.detectMultiScale(gray, 1.3, 5) # Return list of bodys could be captured.
        
        if len(faces) + len(bodys) > 0: # If somebody capture with camera.
            recording = True

        
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
            # frame : image
            # (x, y) , (x + width, y + height) : position of rectangle
            # (255, 0, 0) : color of rectangle
            # 3 : thickness of rectangle is 3 pixel
        for (x, y, width, height) in bodys:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 255), 3)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'): # If i press q on keyboard capture is going to end.
            break
    capture.release() # Release my Camera
    cv2.destroyAllWindows() # Kill

if __name__ == "__main__":
    main() # I Love C, Python 
