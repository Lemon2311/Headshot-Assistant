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




def find_frame_center(frame):
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    return center_x, center_y

def calculate_distance(x1, y1, x2, y2):
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Initialize YOLO model
model = YOLO("yolov8n.pt")

ahk = AHK()
win = ahk.win_get(title='RESIDENT EVIL 2')

# Check if CUDA is available and if so, use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# Initialize the keyboard controller
keyboard = Controller()

# Flag to check if left click is pressed
left_click_pressed = False
right_click_pressed = False

# Mouse callback function for pynput
def on_click(x, y, button, pressed):
    global left_click_pressed
    if button == button.left and pressed:
        left_click_pressed = True
        print("Left click detected")  # Debug message to confirm the click
    global right_click_pressed
    if button == button.right and pressed:
        right_click_pressed = True
        print("Right click detected")    

def aim(coords, increment):
    threshold = 10

    
    # Get the current mouse position
    mouse_position = ahk.get_mouse_position()

    # Calculate the difference in coordinates
    dx = coords[0] - mouse_position[0]
    dy = coords[1] - mouse_position[1]

    # Get a 1 or -1 based on the direction of the difference
    direction_x = 1 if dx > 0 else -1
    direction_y = 1 if dy > 0 else -1

    # If the absolute difference is less than the threshold, break the loop
    if abs(dx) < threshold and abs(dy) < threshold:
        print(coords, mouse_position)

        # Move the mouse in the calculated direction
        ahk.mouse_move(x=increment * direction_x, y=increment * direction_y, blocking=True, speed=0, relative=True)

# Function to start the mouse listener in a separate thread
def start_mouse_listener():
    with Listener(on_click=on_click) as listener:
        listener.join()

# Start the mouse listener thread
mouse_listener_thread = threading.Thread(target=start_mouse_listener)
mouse_listener_thread.daemon = True  # Make sure this thread exits when the main program exits
mouse_listener_thread.start()

# Example usage of record_window_stream, which records the video stream from the desired on-screen window and returns the frames and FPS
for frame, fps in record_window_stream("RESIDENT EVIL 2"):
    # Process the frame and FPS as needed
    print(f"FPS: {fps:.2f}", end='\r')  # Display FPS

    # Convert the frame to RGB (YOLO expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the frame to a smaller size that's still a multiple of 32, while maintaining the aspect ratio
    height, width, _ = frame_rgb.shape
    new_height = 640
    new_width = int(new_height * width / height)
    new_width = (new_width // 32) * 32  # Make sure the new width is a multiple of 32
    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
    
    # Convert the frame to a PyTorch tensor and add a batch dimension
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Convert the tensor to a floating point type
    frame_tensor = frame_tensor.float() / 255.0  # Also normalize the values to [0, 1] range as YOLO expects this

    # Run YOLO on the frame tensor
    results = model.predict(frame_tensor, classes=0, conf=0.35)

    # Extract bounding boxes and iterate through them
    boxes = results[0].boxes.xyxy.tolist()

    centers_list = []

    for box in boxes:
      
        x1, y1, x2, y2 = map(int, box)
        x1 = int(x1 * width / new_width)
        y1 = int(y1 * height / new_height)
        x2 = int(x2 * width / new_width)
        y2 = int(y2 * height / new_height)

        # Calculate center coordinates of the box
        center_x = (x1 + x2) // 2
        center_y = y1+y1*10//100  # Adjust the y-coordinate as needed
        
        radius = 10  # Adjust the radius of the circle as needed
        color = (0, 255, 0)  # BGR color (green in this case)

        # Draw a circle at the center of each detected box
        cv2.circle(frame, (center_x, center_y), radius, color, -1)  # -1 to fill the circle

        centers_list.append((center_x, center_y))
        
    min_distance = 1000000

    for i in range(len(centers_list)-1):
        x1, y1 = centers_list[i]
        x2, y2 = find_frame_center(frame)
        distance = calculate_distance(x1, y1, x2, y2)
        if(distance < min_distance):
            min_distance = distance
            min_distance_coords = (x1, y1)
            #update to keep track of al the distances to center and take the min then shooting

# Get the size of the screen
    screen_width = get_monitors()[0].width
    screen_height = get_monitors()[0].height

    if(right_click_pressed):
        right_click_pressed = False    

# Initialize the controllers
    mouse_controller = MouseController()
    keyboard_controller = KeyboardController()

# Check for left click and simulate pressing 'H'
    if left_click_pressed:
    # Press the right mouse button
        mouse_controller.press(Button.right)

        x, y = min_distance_coords
    # Get the size of the image
        image_width, image_height = frame.shape[1], frame.shape[0]

    # Calculate the scaling factors
        scale_x = screen_width / image_width
        scale_y = screen_height / image_height

    # Scale the coordinates
        x *= 1
        y *= 1

    # Move the mouse
        print('The current pointer position is', ahk.get_mouse_position())
        print('Commanded pos', x, y)
        
        threshold = 10
        increment = 5

        # Get the current mouse position
        mouse_position = ahk.get_mouse_position()

        # Calculate the difference in coordinates
        dx = x - mouse_position[0]
        dy = y - mouse_position[1]

        # Get a 1 or -1 based on the direction of the difference
        direction_x = 1 if dx > 0 else -1
        direction_y = 1 if dy > 0 else -1

        # If the absolute difference is less than the threshold, break the loop
        if not abs(dx) < threshold and abs(dy) < threshold:
            ahk.mouse_move(x=increment * direction_x, y=increment * direction_y, blocking=True, speed=0, relative=True)    
        else:
            time.sleep(0.3)  # Introduce a small delay
            print("Simulating 'H' key press")  # Debug message to confirm the 'H' key press
            keyboard_controller.press('h')
            time.sleep(0.1)  # Introduce a small delay
            keyboard_controller.release('h')
            mouse_controller.release(Button.right)
            left_click_pressed = False  # Reset the flag
            print('The current pointer position is', ahk.get_mouse_position())
            print('Commanded pos', x, y)

        print(x,y, mouse_position)      

    # Display the frame with circles drawn on it
    cv2.circle(frame, find_frame_center(frame), 5, (255,0,0), -1)  # -1 to fill the circle
    cv2.imshow('Frame', frame)
    print(find_frame_center(frame))
    print("min dist:",min_distance)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
