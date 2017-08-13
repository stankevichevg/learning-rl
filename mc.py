import numpy as np
from gridworld import GridWorld


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


def get_return(state_list, gamma):
    """
    Производит расчет сумарного дисконтированного вознаграждения.
    
    :param state_list: история наблюдений
    :param gamma: дисконтирующий множетель
    :return: рассчитанное сумарное дисконтированное вознаграждение
    """
    counter = 0
    return_value = 0
    for visit in state_list:
        reward = visit[-1]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value


def update_policy(episode_list, policy_matrix, state_action_matrix):
    """
    Жадный алгоритм обновления политики. На остнове матрицы состояние-действие выбирается действие
    с максимальным ожидаемым вознаграждением для заданного состояния.
    
    @return обновленная политика действий
    """
    for visit in episode_list:
        observation = visit[0]
        col = observation[1] + (observation[0]*4)
        if policy_matrix[observation[0], observation[1]] != -1:
            policy_matrix[observation[0], observation[1]] = np.argmax(state_action_matrix[:,col])
    return policy_matrix


def mc_prediction(env, policy_matrix, gamma = 0.999, tot_epoch = 50000, print_epoch = 1000):
    """
    Алгоритм пассивного обучения с подкреплением MC Prediction. 
    Имеется политика действий в среде env, заданная матрицей policy_matrix.
    Алгоритм находит вектор ценности состояний при среды при следовании заданной политике (предсказывает ценность состояния).
    
    :param env: среда в которой агент производит действия
    :param policy_matrix: матрица политики действий
    :param gamma: дисконтирующий фактор
    :param tot_epoch: общее число эпох для оценки параметров
    :param print_epoch: число эпох после которых необходимо распечатать текущее состояние оценок ценностей состояний.
    :return: матрицу ценностей состояний
    """

    # Матрица, содержащая сумму ожидаемых вознаграждений, полученных после первого
    # наблюдения состояния для каждого эпизода.
    utility_matrix = np.zeros((3, 4))
    # Матрица с числом эпизодов, в которых состояние встречалось хотя бы один раз.
    running_mean_matrix = np.full((3, 4), 1.0e-10)

    for epoch in range(tot_epoch):
        # Начало нового эпизода. Создаем список для истории эпизода.
        episode_list = list()
        # Сбрасываем состояние среды и получаем начальное состояние.
        observation = env.reset(exploring_starts=False)
        for _ in range(1000):
            # Выбираем действие для текущего состояния согласно имеющейся политике
            action = policy_matrix[observation[0], observation[1]]
            # Совершаем действие, получаем следующее состояние и вознаграждение.
            observation, reward, done = env.step(action)
            # Добавляем наблюдение в историю эпизода
            episode_list.append((observation, reward))
            # Выходим, если попали в терминальное состояние
            if done: break
        # Эпизод закончен. Считаем ценности состояний.
        counter = 0
        # Создаем матрицу индикаторов, что мы уже наблюдали состояние
        checkup_matrix = np.zeros((3, 4))
        # Реализуем First-Visit MC. Для каждого первого наблюдаемого состояния в истории эпизода
        # получает дисконтированное вознаграждение всех последующих наблюдений.
        for visit in episode_list:
            observation = visit[0]
            row = observation[0]
            col = observation[1]
            if checkup_matrix[row, col] == 0:
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                utility_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        # если требуется, выводим матрицу
        if epoch % print_epoch == 0:
            print("Utility matrix after " + str(epoch + 1) + " iterations:")
            print(utility_matrix / running_mean_matrix)

    # рассчитываем и возвращаем результат
    return utility_matrix / running_mean_matrix, tot_epoch


def mc_control(env, policy_matrix, gamma=0.999, tot_epoch=500000, print_epoch=1000):
    """
    Алгоритм пассивного обучения с подкреплением MC Control или 
    обопщенный Policy Iteration алгоритм (GPI).
    Основное отличие от MC Prediction в том, что в начале не задана политика, она также 
    находится в результате работы алгоритма. При этом используются оценки ценности действий 
    в заданных состояних (Q-функция). В конце каждой эпохи по жадному правилу происходит обновление 
    политики в соответсвии с текущим состоянием Q-функции.
    
    :param env: среда в которой агент производит действия
    :param policy_matrix: матрица политики действий
    :param gamma: дисконтирующий фактор
    :param tot_epoch: общее число эпох для оценки параметров
    :param print_epoch: число эпох после которых необходимо распечатать текущее состояние оценок ценностей состояний.
    :return: матрицу ценностей действий для состояний (Q)
    """

    # Матрица с числом эпизодов, в которых состояние встречалось хотя бы один раз.
    running_mean_matrix = np.full((4,12), 1.0e-10)
    # Случайная матрица состояние-действие (табличная апроксимация Q-функции)
    state_action_matrix = np.random.random_sample((4, 12))

    for epoch in range(tot_epoch):
        # Начало нового эпизода. Создаем список для истории эпизода.
        episode_list = list()
        # Сбрасываем состояние среды и получаем начальное состояние.
        observation = env.reset(exploring_starts=True)
        is_starting = True
        for _ in range(1000):
            # Выбираем действие для текущего состояния согласно имеющейся политике
            action = policy_matrix[observation[0], observation[1]]
            # Если мы в начале эпизода, то случайно равновероятно выбираем первое действие
            if is_starting:
                action = np.random.randint(0, 4)
                is_starting = False
            # Совершаем действие, получаем следующее состояние и вознаграждение.
            new_observation, reward, done = env.step(action)
            # Добавляем наблюдение в историю эпизода
            episode_list.append((observation, action, reward))
            observation = new_observation
            if done:
                break
        # Эпизод закончен. Считаем ценности действий в состояниях (Q).
        counter = 0
        # Создаем матрицу индикаторов, что мы уже наблюдали состояние
        checkup_matrix = np.zeros((4, 12))
        # Реализуем First-Visit MC. Для каждого первого наблюдаемого состояния в истории эпизода
        # получает дисконтированное вознаграждение всех последующих наблюдений.
        # 1 - Шаг оценки в GPI.
        for visit in episode_list:
            observation = visit[0]
            action = visit[1]
            col = int(observation[1] + (observation[0] * 4))
            row = int(action)
            if checkup_matrix[row, col] == 0:
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                state_action_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        # 1 - Шаг обновления в GPI.
        policy_matrix = update_policy(episode_list, policy_matrix, state_action_matrix / running_mean_matrix)
        # если требуется, выводим промежуточный результат работы алгоритма
        if (epoch % print_epoch == 0):
            print("")
            print("State-Action matrix after " + str(epoch + 1) + " iterations:")
            print(state_action_matrix / running_mean_matrix)
            print("Policy matrix after " + str(epoch + 1) + " iterations:")
            print(policy_matrix)
    # Возвращаем Q-функцию
    return state_action_matrix / running_mean_matrix, tot_epoch


def main():

    # Определяем матрицу политики действий, определяет в каком состоянии какое действие нужно совершать.
    # 0 - Вверх,
    # 1 - Вправо,
    # 2 - Вниз,
    # 3 - Влево,
    # NaN - Не определено,
    # -1 - Нет действия
    # та самая оптимальная политика
    policy_matrix = np.array([[1, 1,      1, -1],
                              [0, np.NaN, 0, -1],
                              [0, 3,      3, 3]])
    env = create_env()
    utility, tot_epoch = mc_prediction(env, policy_matrix)
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility)

    # Случайная матрица политики действий
    policy_matrix = np.random.randint(low=0, high=4,
                                      size=(3, 4)).astype(np.float32)
    policy_matrix[1, 1] = np.NaN
    policy_matrix[0, 3] = policy_matrix[1, 3] = -1

    env = create_env()
    q, tot_epoch = mc_control(env, policy_matrix)
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(q)


if __name__ == "__main__":
    main()
