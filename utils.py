import numpy as np


def print_policy(p, shape):
    """
    Распечатывает политику.

    Использует следующие обозначения:
    ^ вверх
    v вниз
    < влево
    > вправо
    * терминальное состояние
    # действие не может быть определено (скорее всего недостижимое состояние)
    """
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)