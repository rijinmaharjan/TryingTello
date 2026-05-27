import cv2
import time
import urllib.request
import os
from djitellopy import Tello
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- NEW MEDIAPIPE ASSET SETUP ---
model_path = 'hand_landmarker.task'
model_url = "[https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)"

if not os.path.exists(model_path):
    print("Downloading MediaPipe Hand Landmarker asset... please wait.")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete!")

# Initialize the new Hand Landmarker Task API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- INITIALIZE YOLO & TELLO ---
model = YOLO('yolov8n.pt') 
drone = Tello()
drone.connect()
print(f"Battery Life: {drone.get_battery()}%")

drone.streamon()
frame_read = drone.get_frame_read()

print("Taking off in 3 seconds... Stand back!")
time.sleep(3)
drone.takeoff()
drone.send_rc_control(0, 0, 10, 0) 
time.sleep(2)

spinning_mode = False
spin_start_time = 0

def analyze_hand_gesture_modern(landmarks):
    """
    New MediaPipe formats landmarks as objects inside a list.
    Tip indices: Index(8), Middle(12), Ring(16), Pinky(20)
    Knuckle indices: Index(6), Middle(10), Ring(14), Pinky(18)
    """
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    
    fingers_open = 0
    for t, k in zip(tips, knuckles):
        if landmarks[t].y < landmarks[k].y:
            fingers_open += 1
            
    return "OPEN" if fingers_open >= 3 else "FIST"

try:
    while True:
        img = frame_read.frame
        if img is None:
            continue
            
        img = cv2.resize(img, (640, 480))
        h, w, _ = img.shape
        center_x = w // 2
        
        rc_up_down = 0
        rc_yaw = 0
        
        # --- SPINNING MODE ---
        if spinning_mode:
            if time.time() - spin_start_time < 3.0:
                drone.send_rc_control(0, 0, 0, 60)
                cv2.putText(img, "Executing 360 Spin Search...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("Tello Workshop Control Panel", img)
                if cv2.waitKey(1) & 0xFF == 27: break
                continue
            else:
                spinning_mode = False
                drone.send_rc_control(0, 0, 0, 0)
        
        # --- YOLO PERSON DETECTION ---
        results = model(img, classes=[0], verbose=False)
        for r in results[0].boxes:
            box = r.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            person_center_x = (x1 + x2) // 2
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img, (center_x, 0), (center_x, h), (255, 0, 0), 1)
            cv2.line(img, (person_center_x, y1), (person_center_x, y2), (0, 0, 255), 2)
            
            error_x = person_center_x - center_x
            if abs(error_x) > 40:
                rc_yaw = int(error_x * 0.25)
                rc_yaw = max(-70, min(70, rc_yaw))
            break 
            
        # --- NEW MEDIAPIPE GESTURE DETECTION ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            detected_gestures = []
            hand_positions_y = []
            hand_positions_x = []
            
            for hand_landmarks in detection_result.hand_landmarks:
                gesture = analyze_hand_gesture_modern(hand_landmarks)
                detected_gestures.append(gesture)
                
                wrist = hand_landmarks[0]
                hand_positions_y.append(wrist.y)
                hand_positions_x.append(wrist.x)
                
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
            
            num_hands = len(detected_gestures)
            
            # Emergency Land
            if num_hands == 2 and hand_positions_y[0] < 0.3 and hand_positions_y[1] < 0.3:
                cv2.putText(img, "EMERGENCY LAND ACTIVATED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                drone.land()
                break
                
            # 360 Spin Search
            elif num_hands == 2 and abs(hand_positions_x[0] - hand_positions_x[1]) > 0.6:
                spinning_mode = True
                spin_start_time = time.time()
                continue
                
            # Throttle commands
            elif num_hands == 1:
                if detected_gestures[0] == "OPEN":
                    rc_up_down = 25
                    cv2.putText(img, "COMMAND: ASCEND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif detected_gestures[0] == "FIST":
                    rc_up_down = -25
                    cv2.putText(img, "COMMAND: DESCEND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not spinning_mode:
            drone.send_rc_control(0, 0, rc_up_down, rc_yaw)
            
        cv2.imshow("Tello Workshop Control Panel", img)
        if cv2.waitKey(1) & 0xFF == 27:
            drone.land()
            break

except Exception as e:
    print(f"Error: {e}")
    try: drone.land()
    except: pass
finally:
    # Explicit close to bypass Windows C++ memory cleanup error
    try:
        detector.close()
        print("MediaPipe tracking engine closed cleanly.")
    except:
        pass
    drone.streamoff()
    cv2.destroyAllWindows()
    print("System offline.")