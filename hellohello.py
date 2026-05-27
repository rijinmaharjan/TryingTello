import cv2
import numpy as np
from djitellopy import Tello

# --- CONFIGURATION CONSTANTS ---
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# Face tracking targets
TARGET_FACE_AREA = 50000 
AREA_THRESHOLD = 8000  

# Tracking gains (PID-ish tuning)
PID_X = 0.25  # Controls Rotation (Yaw)
PID_Y = 0.35  # Controls Altitude (Up/Down)
PID_Z = 0.0015 # Controls Depth (Forward/Backward)

def main():
    # Initialize and connect to Tello
    me = Tello()
    me.connect()
    print(f"Battery Level: {me.get_battery()}%")
    
    # Initialize velocity attributes
    me.for_back_velocity = 0
    me.left_right_velocity = 0
    me.up_down_velocity = 0
    me.yaw_velocity = 0

    # Start Video Stream
    me.streamoff()
    me.streamon()
    frame_read = me.get_frame_read()

    # Load OpenCV Pre-trained Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    tracking_mode = False
    print("\n--- NEW STABLE CONTROLS ---")
    print("Takeoff: SPACEBAR  | Land: BACKSPACE")
    print("Move Horizontal: W (forward), S (back), A (left), D (right)")
    print("Altitude & Yaw:  I (up), K (down), J (turn left), L (turn right)")
    print("Toggle Face Tracking: T")
    print("Quit Program: ESC\n")

    while True:
        # 1. Capture frame and fix orientation
        frame = frame_read.frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw static center crosshair lines
        cv2.line(frame, (CENTER_X - 20, CENTER_Y), (CENTER_X + 20, CENTER_Y), (255, 255, 255), 1)
        cv2.line(frame, (CENTER_X, CENTER_Y - 20), (CENTER_X, CENTER_Y + 20), (255, 255, 255), 1)

        # 2. Face Detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        
        # Reset velocities every loop if tracking is active (recalculated below)
        if tracking_mode:
            me.left_right_velocity = 0
            me.for_back_velocity = 0
            me.up_down_velocity = 0
            me.yaw_velocity = 0

        if len(faces) > 0:
            # Focus on the largest face (closest to camera)
            largest_face = max(faces, key=lambda b: b[2] * b[3])
            x, y, w, h = largest_face
            
            face_center_x = x + (w // 2)
            face_center_y = y + (h // 2)
            face_area = w * h

            # Visual Overlays (Lines and Box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (CENTER_X, CENTER_Y), (face_center_x, face_center_y), (0, 0, 255), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

            # 3. Autonomous Tracking Logic
            if tracking_mode:
                error_x = face_center_x - CENTER_X
                error_y = CENTER_Y - face_center_y 
                error_area = TARGET_FACE_AREA - face_area

                # Calculate Multi-directional velocities
                me.yaw_velocity = int(np.clip(error_x * PID_X, -50, 50))
                me.up_down_velocity = int(np.clip(error_y * PID_Y, -40, 40))
                
                if abs(error_area) > AREA_THRESHOLD:
                    me.for_back_velocity = int(np.clip(error_area * PID_Z, -35, 35))
                else:
                    me.for_back_velocity = 0
                    
                me.left_right_velocity = 0 

        # Display UI labels
        status_text = f"Tracking Mode: {'ACTIVE' if tracking_mode else 'MANUAL'}"
        color = (0, 255, 0) if tracking_mode else (0, 165, 255)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Battery: {me.get_battery()}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Render output stream window
        cv2.imshow("Tello DJI Camera Feed", frame)

        # 4. Keyboard Listener
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC Key
            break
        elif key == ord('t'):  # Toggle tracking
            tracking_mode = not tracking_mode
            if not tracking_mode:
                me.send_rc_control(0, 0, 0, 0)
        elif key == ord(' '):  # SPACE to Take off
            me.takeoff()
        elif key == ord('\b') or key == ord('b'):  # Backspace or 'B' to Land
            me.land()
            
        # 5. Process Manual Control (Only when tracking is OFF)
        if not tracking_mode:
            lr, fb, ud, yv = 0, 0, 0, 0
            speed = 50 
            
            # WASD Controls (Horizontal plane translation)
            if key == ord('w'): fb = speed
            elif key == ord('s'): fb = -speed
            if key == ord('a'): lr = -speed
            elif key == ord('d'): lr = speed
            
            # IJKL Controls (Cross-platform safe replacement for Arrow keys)
            if key == ord('i'): ud = speed       # Up
            elif key == ord('k'): ud = -speed   # Down
            if key == ord('j'): yv = -speed     # Turn Left
            elif key == ord('l'): yv = speed      # Turn Right
            
            me.send_rc_control(lr, fb, ud, yv)
            
        elif tracking_mode:
            # Active transmission of calculated tracking speeds
            me.send_rc_control(me.left_right_velocity, me.for_back_velocity, 
                               me.up_down_velocity, me.yaw_velocity)

    # Cleanup Routine on Exit
    me.land()
    me.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()