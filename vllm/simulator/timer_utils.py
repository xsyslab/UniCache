# SPDX-License-Identifier: Apache-2.0
# timer_utils.py
class VirtualTimer:

    def __init__(self, start_time=0):
        self.start_time = start_time
        self.current_time = start_time

    def advance(self, delta_time):
        self.current_time += delta_time

    def get_current_time(self):
        return self.current_time

    def set_time(self, time):
        self.current_time = time


GLOBAL_TIMER = VirtualTimer()
