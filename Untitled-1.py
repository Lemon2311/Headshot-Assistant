from ahk import AHK
import time
ahk = AHK()
time.sleep(5)
ahk.win_get(title='RESIDENT EVIL 2')
print(ahk.get_mouse_position())