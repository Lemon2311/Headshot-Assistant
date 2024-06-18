from ahk import AHK
import time

# Initialize an AHK object
ahk = AHK()

# Delay execution for 5 seconds (you might want to perform some initial setup here)
time.sleep(5)

# Get the window with the title 'RESIDENT EVIL 2'
win = ahk.win_get(title='RESIDENT EVIL 2')

def aim(coords, increment):
    threshold = 10

    while True:
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
            break

        # Move the mouse in the calculated direction
        ahk.mouse_move(x=increment * direction_x, y=increment * direction_y, blocking=True, speed=0, relative=True)

aim((300, 300), 5)  # Example usage
