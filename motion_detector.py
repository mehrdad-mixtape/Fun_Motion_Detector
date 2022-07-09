import cv2
import time, datetime

def main():
    # ------------------------------Initial Camera------------------------------
    capture = cv2.VideoCapture(0) # Access to my camera I have one camera and index is 0.
    
    # ------------------------------Load Face & Body Detection------------------------------
    face_cc = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    body_cc = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')

    # ------------------------------Define Detecting Variables------------------------------
    detection = False
    detection_stopped_time = None
    timer_started = False
    SECOND_TO_RECORD_AFTER_DETEDCTION = 10

    # ------------------------------Define Recording Variables------------------------------
    frame_size = (int(capture.get(3)), int(capture.get(4)))
    frame_rate = 10.0
    video_format = cv2.VideoWriter_fourcc(*'mp4v')

    # ------------------------------Start------------------------------
    while True:
        _, frame = capture.read() # Return _ : somethings , frame : frames of video.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Get gray color to all frames. COLOR_BGR2GRAY : get gray color
        faces = face_cc.detectMultiScale(gray, 10.0, 5) # Return list of faces could be captured.
        # 1.3 : accurcy and speed of algorithm (lower is faster).
        # 5 : how many face can detect? 5 is middle (lower than 5 decrease power detect / more than 5 increase power detect)
        bodys = body_cc.detectMultiScale(gray, 10.0, 5) # Return list of bodys could be captured.
        
        if len(faces) + len(bodys) > 0: # If somebody be in camera zone.
            if detection == True: # Continue recording
                timer_started = False
                print('Continue Recording ...')

            else:
                detection = True # Start recording
                video_output = cv2.VideoWriter(
                    f"video_{datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.mp4",
                    video_format,
                    frame_rate,
                    frame_size
                )
                print('Started Recording ...')
        
        elif detection == True: # Wait for somebody to comeback to camera zone.
            if timer_started:
                if time.time() - detection_stopped_time >= SECOND_TO_RECORD_AFTER_DETEDCTION: # After 5 second somebody don't comeback to camera zone, stop recording.
                    detection = False
                    timer_started = False
                    video_output.release()
                    print('Stop Recording ...')

            else:
                timer_started = True
                detection_stopped_time = time.time()
            
        if detection == True:
            video_output.write(frame) # Write frames on video

        if cv2.waitKey(1) == ord('q'): # If i press q on keyboard capture is going to end.
            video_output.release()
            break
        
        cv2.imshow("Camera", frame) # Show you window camera

    video_output.release() # Save and release video
    capture.release() # Release my Camera
    cv2.destroyAllWindows() # Kill

if __name__ == "__main__":
    main()
