import cv2
import numpy as np
from djitellopy import Tello
import time

# --- Configuration Constants ---
WIDTH, HEIGHT = 480, 360  # Slightly larger frame for better clicking accuracy
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2

X_DEADZONE = 35
Y_DEADZONE = 30
TARGET_AREA = 8000  # Adjusted for larger frame resolution
AREA_DEADZONE = 1500
MAX_SPEED = 25

# --- Global UI Variables ---
selected_box = None      # Stores the coordinates of the manually selected person [x, y, w, h]
is_flying = False
drone = None

# Button coordinates [x1, y1, x2, y2]
BTN_TAKEOFF = [10, 10, 110, 45]
BTN_LAND    = [120, 10, 220, 45]
BTN_STOP    = [230, 10, 330, 45]

def mouse_click_handler(event, mouse_x, mouse_y, flags, param):
    """Handles user clicks on the video window for selecting targets and buttons."""
    global selected_box, is_flying, drone

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Check if user clicked UI Buttons
        # Takeoff Button
        if BTN_TAKEOFF[0] <= mouse_x <= BTN_TAKEOFF[2] and BTN_TAKEOFF[1] <= mouse_y <= BTN_TAKEOFF[3]:
            if not is_flying and drone:
                print("UI Action: Taking off...")
                drone.takeoff()
                drone.move_up(40)
                is_flying = True
            return

        # Land Button
        if BTN_LAND[0] <= mouse_x <= BTN_LAND[2] and BTN_LAND[1] <= mouse_y <= BTN_LAND[3]:
            if is_flying and drone:
                print("UI Action: Landing...")
                drone.land()
                is_flying = False
            return

        # Emergency Stop Button (Forces hovering state)
        if BTN_STOP[0] <= mouse_x <= BTN_STOP[2] and BTN_STOP[1] <= mouse_y <= BTN_STOP[3]:
            print("UI Action: Emergency Stop Tracking (Hovering)")
            selected_box = None # Clears target tracking instantly
            if is_flying and drone:
                drone.send_rc_control(0, 0, 0, 0)
            return

        # 2. Check if user clicked a person/face
        # 'param' passes the currently detected faces list from the main loop
        detected_faces = param
        for (x, y, w, h) in detected_faces:
            if x <= mouse_x <= (x + w) and y <= mouse_y <= (y + h):
                selected_box = [x, y, w, h]
                print(f"Target Acquired at position X: {x}, Y: {y}")
                return

def draw_ui_overlay(frame, faces):
    """Draws interactive buttons and tracking indicators on the frame."""
    # Draw Takeoff Button (Greenish)
    cv2.rectangle(frame, (BTN_TAKEOFF[0], BTN_TAKEOFF[1]), (BTN_TAKEOFF[2], BTN_TAKEOFF[3]), (0, 180, 0), -1)
    cv2.putText(frame, "TAKEOFF", (BTN_TAKEOFF[0]+12, BTN_TAKEOFF[1]+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw Land Button (Red)
    cv2.rectangle(frame, (BTN_LAND[0], BTN_LAND[1]), (BTN_LAND[2], BTN_LAND[3]), (0, 0, 200), -1)
    cv2.putText(frame, "LAND", (BTN_LAND[0]+28, BTN_LAND[1]+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw Stop Button (Orange)
    cv2.rectangle(frame, (BTN_STOP[0], BTN_STOP[1]), (BTN_STOP[2], BTN_STOP[3]), (0, 100, 255), -1)
    cv2.putText(frame, "STOP/HOVER", (BTN_STOP[0]+8, BTN_STOP[1]+23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    # Draw passive bounding boxes around all available people/faces
    for (x, y, w, h) in faces:
        # Check if this specific face is the currently selected target
        if selected_box and abs(x - selected_box[0]) < 30 and abs(y - selected_box[1]) < 30:
            continue # Skip drawing passive box over active target
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1) # Yellow box for unselected
        cv2.putText(frame, "Click to Follow", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def main():
    global selected_box, is_flying, drone
    
    # Initialize detector and drone
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
    drone.streamon()
    time.sleep(2)

    # Setup OpenCV Window and register our click handler
    window_name = "DJI Tello Smart UI Control"
    cv2.namedWindow(window_name)
    
    # This list will hold faces contextually so the mouse handler can read them
    current_faces = []
    cv2.setMouseCallback(window_name, mouse_click_handler, param=current_faces)

    while True:
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        if frame is None:
            continue
            
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect all faces in current view
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        
        # Update our tracking reference list for the mouse click callback
        current_faces.clear()
        current_faces.extend(faces)

        # Movement speeds variables
        yaw_speed = 0
        up_down_speed = 0
        forward_backward_speed = 0
        
        target_found = False

        # If a target has been clicked/selected previously, look for it
        if selected_box is not None:
            best_match = None
            min_dist = 9999
            
            # Match the selected box to the newly updated face positions
            for (x, y, w, h) in faces:
                # Calculate distance from our tracked coordinate to find the same person
                dist = np.sqrt((x - selected_box[0])**2 + (y - selected_box[1])**2)
                if dist < min_dist and dist < 60: # Threshold to ensure it's the same moving target
                    min_dist = dist
                    best_match = [x, y, w, h]
            
            if best_match:
                target_found = True
                selected_box = best_match # Lock on to updated position
                x, y, w, h = best_match
                
                # Draw Lock-On UI Graphics
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) # Heavy green box
                cv2.putText(frame, "TARGET LOCKED", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Control loops error calculation
                face_center_x = x + (w // 2)
                face_center_y = y + (h // 2)
                face_area = w * h
                
                error_x = face_center_x - CENTER_X
                error_y = CENTER_Y - face_center_y
                
                # Calculate commands if airborne
                if is_flying:
                    if abs(error_x) > X_DEADZONE:
                        yaw_speed = int(np.clip(0.18 * error_x, -MAX_SPEED, MAX_SPEED))
                    if abs(error_y) > Y_DEADZONE:
                        up_down_speed = int(np.clip(0.18 * error_y, -MAX_SPEED, MAX_SPEED))
                    if abs(face_area - TARGET_AREA) > AREA_DEADZONE:
                        area_error = TARGET_AREA - face_area
                        forward_backward_speed = int(np.clip(0.0035 * area_error, -MAX_SPEED, MAX_SPEED))
            else:
                # Target stepped out of frame or visibility lost
                cv2.putText(frame, "TARGET LOST - HOVERING", (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                selected_box = None # Lose lock

        # Apply speeds to drone
        if is_flying:
            drone.send_rc_control(0, forward_backward_speed, up_down_speed, yaw_speed)

        # Draw UI overlay blocks (Buttons and unselected user indicator boxes)
        draw_ui_overlay(frame, faces)
        
        # Display window
        cv2.imshow(window_name, frame)
        
        # Manual fallback keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if is_flying:
                drone.land()
            break

    drone.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()