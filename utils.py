import numpy as np

from gridworld import GridWorld


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


def create_env():
    """
    Создает среду для экспериментов

    :return: среду
    """
    # Создаем среду в виде сетки 3x4
    env = GridWorld(3, 4)
    # Задаем матрицу состояний
    state_matrix = np.zeros((3, 4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    # Задаем матрицу вознаграждений
    # Для всех кроме терминальных состояний вознаграждение -0.04
    reward_matrix = np.full((3, 4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    # Задаем матрицу вероятности совершения действия
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    # Настраиваем и возвращаем среду
    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    return env