# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:31:21 2019

@author: JC056596
"""

import numpy as np

class Neuron(object):
    def __init__(self, inputs, weights = "random", firing_mode = "gaussian"):
        self.inputs = inputs
        if weights == "random":
            self.weights = np.random.rand(len(self.inputs))
        else:
            self.weights = np.array(weights)
        
        self.firing_mode = firing_mode
        
        self.reset_firing_value()
        
    def reset_firing_value(self):
        self.fired = False
        self.firing_value = 0
    
    def fire(self, inputs = None):
        if self.fired:
            return self.firing_value
        else:
            self.fired = True
            
        
        input_values = tuple(neuron.fire(inputs) for neuron in self.inputs) if self.inputs != None else inputs
        self.sum_inputs = np.dot(input_values, self.weights)
        
        if self.firing_mode == "gaussian":
            self.firing_value = self.fire_gaussian()
        else:
            self.firing_value = self.fire_default()
        return self.firing_value
    
    def fire_gaussian(self):
        return 1 - np.exp(-self.sum_inputs*self.sum_inputs)
    
    def fire_default(self):
        if self.sum_inputs > 0:
            return 1
        elif self.sum_inputs == 0:
            return 0
        else:
            return -1

class NeuralNet(object):
    def __init__(self, inner_neurons, output_neuron):
        self.inner_neurons = inner_neurons
        self.output_neuron = output_neuron
        
    def calculate(self, inputs):
        return self.output_neuron.fire(inputs)

class LayeredNet(NeuralNet):
    