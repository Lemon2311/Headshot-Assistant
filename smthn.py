from pynput.mouse import Button, Controller
from screeninfo import get_monitors

# Initialize the mouse controller
mouse_controller = Controller()

# Get the size of the screen
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

# Calculate the center of the screen
center_x = screen_width / 2
center_y = screen_height / 2

# Move the mouse to the center of the screen
mouse_controller.position = (center_x, center_y)