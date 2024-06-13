import torch
import cv2
from ultralytics import YOLO
from zest import record_window_stream
from pynput.keyboard import Controller
from pynput.mouse import Listener
import threading
import time

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Check if CUDA is available and if so, use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize the keyboard controller
keyboard = Controller()

# Flag to check if left click is pressed
left_click_pressed = False

# Mouse callback function for pynput
def on_click(x, y, button, pressed):
    global left_click_pressed
    if button == button.left and pressed:
        left_click_pressed = True
        print("Left click detected")  # Debug message to confirm the left click

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
    new_height = 320
    new_width = int(new_height * width / height)
    new_width = (new_width // 32) * 32  # Make sure the new width is a multiple of 32
    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
    
    # Convert the frame to a PyTorch tensor and add a batch dimension
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Convert the tensor to a floating point type
    frame_tensor = frame_tensor.float() / 255.0  # Also normalize the values to [0, 1] range as YOLO expects this

    # Run YOLO on the frame tensor
    results = model.predict(frame_tensor, show=True)

    # Check for left click and simulate pressing 'H'
    if left_click_pressed:
        print("Simulating 'H' key press")  # Debug message to confirm the 'H' key press
        keyboard.press('h')
        time.sleep(0.1)  # Introduce a small delay
        keyboard.release('h')
        left_click_pressed = False  # Reset the flag
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
