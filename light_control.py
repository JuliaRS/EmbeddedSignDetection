import RPi.GPIO as GPIO
import numpy as np


class LightContoller:
    def __init__(self):
        """
        Three pins with different resistances are available. The less resistance, the brighter the lamp.
        Different pins can be joined in parallel combinations.
        """
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.pins = [18, 17, 15]  # + [14] (connected, but does nothing)
        GPIO.setup(self.pins, GPIO.OUT)

        epsilon = 0.1  # wires resistance
        pin_resistances = np.array([27 + epsilon, 18 + epsilon, epsilon])  # lamp brightness depends on resistance
        self.pin_combinations = [[], [0], [1], [0, 1], [2]]  # from darkest to brightest
        self.pin_combinations = [np.array(combination) for combination in self.pin_combinations]
        self.resistances = [1. / np.sum(1./pin_resistances[np.array(combination)])
                            for combination in self.pin_combinations[1:]]
        self.resistances = [np.inf] + self.resistances

        self.current_position = 0  # lamp turned off
        self.last_index = -1
        self.turn_off_indexes = [3, 4, 14, 19]
        self.turn_on_indexes = [5, 6, 20]
        self.brighter_indexes = [12, 13, 17, 18, 23, 24]
        self.darker_indexes = [10, 11, 15, 16, 25, 26]
        self.turn_off()  # clear old state

    def turn_on(self):
        print("Turning the lamp on")
        self.current_position = len(self.pin_combinations) - 2
        if len(self.pin_combinations[self.current_position]) != 0:
            return
        self.apply_pin_combination(self.pin_combinations[-2])  # turn the brightest

    def turn_off(self):
        print("Turning the lamp off")
        self.current_position = 0
        self.apply_pin_combination(self.pin_combinations[self.current_position])
        if len(self.pin_combinations[self.current_position]) == 0:
            print("Lamp is already off")
            return

    def step_brighter(self):
        print("Brightening the lamp")
        if self.current_position == len(self.pin_combinations) - 1:
            print("Lamp is already the brightest")
        else:
            self.current_position += 1
            self.apply_pin_combination(self.pin_combinations[self.current_position])

    def step_darker(self):
        print("Darkening the lamp")
        if self.current_position == 0:
            print("Lamp is already turned off")
        else:
            self.current_position -= 1
            self.apply_pin_combination(self.pin_combinations[self.current_position])

    def encode_gesture(self, index):
        if index == self.last_index:  # wait for the end of gesture
            return
        if index in self.turn_off_indexes:
            self.turn_off()
        elif index in self.turn_on_indexes:
            self.turn_on()
        elif index in self.brighter_indexes:
            self.step_brighter()
        elif index in self.darker_indexes:
            self.step_darker()
        self.last_index = index

    def apply_pin_combination(self, pin_combination):
        for pin_number in range(len(self.pins)):
            if pin_number not in pin_combination:
                GPIO.output(self.pins[pin_number],  GPIO.LOW)
            else:
                GPIO.output(self.pins[pin_number],  GPIO.HIGH)

    def shutdown(self):
        GPIO.cleanup()
