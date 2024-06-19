from ahk import AHK
import torch
import cv2
from ultralytics import YOLO
from zest import record_window_stream
from pynput.keyboard import Controller
from pynput.mouse import Listener, Button
import threading
import time
import math
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController
from screeninfo import get_monitors
import sys
import keyboard as key

def find_frame_center(frame):
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    return center_x, center_y

def calculate_distance(x1, y1, x2, y2):
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def on_click(x, y, button, pressed):
    global left_click_pressed
    global animateing
    if button == button.left and pressed:
        left_click_pressed = True
        animateing = True
        print("Left click detected")
    global right_click_pressed
    if button == button.right and pressed:
        right_click_pressed = True
        print("Right click detected")    

def aim(coords, increment, threshold):
    
    mouse_position = ahk.get_mouse_position()

    dx = coords[0] - mouse_position[0]
    dy = coords[1] - mouse_position[1]
    
    print('coords:', coords[0], coords[1])
    print('dx:', dx, 'dy:', dy)

    direction_x = 1 if dx > 0 else -1
    direction_y = 1 if dy > 0 else -1
    
    print('direction_x:', direction_x, 'direction_y:', direction_y)

    if abs(dx) < threshold and abs(dy) < threshold:
        print(coords, mouse_position)

    ahk.mouse_move(x=increment * direction_x, y=increment * direction_y, blocking=True, speed=0, relative=True)

    return dx, dy

def start_mouse_listener():
    with Listener(on_click=on_click) as listener:
        listener.join()


model = YOLO("yolov8n.pt")
ahk = AHK()
win = ahk.win_get(title='RESIDENT EVIL 2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

keyboard = Controller()

left_click_pressed = False
animateing = False
right_click_pressed = False

mouse_listener_thread = threading.Thread(target=start_mouse_listener)
mouse_listener_thread.daemon = True
mouse_listener_thread.start()

for frame, fps in record_window_stream("RESIDENT EVIL 2"):
    
    if key.is_pressed('k'):
        sys.exit()

    print(f"FPS: {fps:.2f}", end='\r')

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    height, width, _ = frame_rgb.shape
    new_height = 640
    new_width = int(new_height * width / height)
    new_width = (new_width // 32) * 32
    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
    
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    frame_tensor = frame_tensor.float() / 255.0  # normalize to [0, 1]

    results = model.predict(frame_tensor, classes=0, conf=0.35)

    boxes = results[0].boxes.xyxy.tolist()

    centers_list = []

    for box in boxes:
      
        x1, y1, x2, y2 = map(int, box)
        x1 = int(x1 * width / new_width)
        y1 = int(y1 * height / new_height)
        x2 = int(x2 * width / new_width)
        y2 = int(y2 * height / new_height)

        center_x = (x1 + x2) // 2
        desired_y = y1+y1*10//100
        
        radius = 10
        color = (0, 255, 0)

        cv2.circle(frame, (center_x, desired_y), radius, color, -1)  # -1 to fill the circle

        centers_list.append((center_x, desired_y))
        
    min_distance = 1000000

    for i in range(len(centers_list)-1):
        x1, y1 = centers_list[i]
        x2, y2 = find_frame_center(frame)
        distance = calculate_distance(x1, y1, x2, y2)
        if(distance < min_distance):
            min_distance = distance
            min_distance_coords = (x1, y1)

    if(right_click_pressed):
        right_click_pressed = False    

    mouse_controller = MouseController()
    keyboard_controller = KeyboardController()

    if left_click_pressed:
        mouse_controller.press(Button.right)        
        if(animateing):
            time.sleep(0.3) # delay to allow the draw animation
            animateing = False

        x, y = min_distance_coords

        print('The current pointer position is', ahk.get_mouse_position())
        print('Commanded pos', x, y)
        
        threshold = 39
        increment = 26
        
        mouse_position = ahk.get_mouse_position()

        screen_centerX, screenCenterY = find_frame_center(frame)

        dx = x - screen_centerX
        dy = y - screenCenterY

        if(dx > threshold):
            x1 = -increment
        else:
            x1 = increment
        
        if(dy < threshold):
            y1 = -increment            
        else:
            y1 = increment

        ahk.mouse_move(x=x1, y=y1, blocking=True, speed=2, relative=True)
        print(x,y,"to",screen_centerX,screenCenterY)

        if abs(dx) < threshold and abs(dy) < threshold:
            print(min_distance_coords, ahk.get_mouse_position())
            print("Simulating 'H' key press")
            keyboard_controller.press('h')
            time.sleep(0.1)
            keyboard_controller.release('h')
            mouse_controller.release(Button.right)
            left_click_pressed = False
            print('The current pointer position is', ahk.get_mouse_position())
            print('Commanded pos', x, y)       

    cv2.circle(frame, find_frame_center(frame), 5, (255,0,0), -1)  # -1 to fill the circle
    cv2.imshow('Frame', frame)
    print(find_frame_center(frame))
    print("min dist:",min_distance)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
