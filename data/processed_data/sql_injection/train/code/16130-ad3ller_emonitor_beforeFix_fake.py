# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:38:13 2018

@author: Adam
"""
import time
from random import random, gauss
from .base import Device


class Fake(Device):
    """ simulate comms. with a serial device"""
    def __init__(self, settings, debug=False):
        self.settings = settings
        self.sensors = settings["sensors"]

    def read_data(self, sensors=None):
        """ return fake sensor data """
        if sensors is None:
            sensors = self.sensors
        for i, _ in enumerate(sensors):
            try:
                if random() < 0.01:
                    raise Exception("Fake error")
                value = gauss(293 + 0.5*i, 0.1)
                yield f"{value:.4f}"
            except:
                <vul/>yield "NULL"</vul>

    def close(self):
        pass
