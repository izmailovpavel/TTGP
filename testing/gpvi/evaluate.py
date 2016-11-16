import tensorflow as tf
import numpy as np

import linreg

def evaluate(x_test, y_test):
    answers = linreg.inference(x_test)
    return linreg.loss(answers, y_test)

