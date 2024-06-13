import torch
import cv2
from ultralytics import YOLO
from zest import capture_and_display, record_window_stream

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Check if CUDA is available and if so, use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()