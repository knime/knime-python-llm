# <<< INJECT THIS AT THE START
import faulthandler
import signal

faulthandler.register(signal.SIGTERM, chain=True)
# >>> END INJECT

import time

def my_func():
  while True:
    print(".")
    time.sleep(1)

my_func()
