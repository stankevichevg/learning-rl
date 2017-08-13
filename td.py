import numpy as np

from utils import create_env


def update_utility(utility_matrix, observation, new_observation, reward, alpha, gamma):
    """
    Обновляет матрицу ценностей состояний.

    @param utility_matrix предыдущее состояние матрицы
    @param observation состояние в момент t
    @param new_observation состояние в момент t+1
    @param reward вознаграждение, полученное после совершения действия
    @param alpha скорость обучения
    @param gamma дисконтирующий фактор
    @return обновленная матрица ценностей состояний
    """
    u = utility_matrix[observation[0], observation[1]]
    u_t1 = utility_matrix[new_observation[0], new_observation[1]]
    utility_matrix[observation[0], observation[1]] += alpha * (reward + gamma * u_t1 - u)
    return utility_matrix


def td_prediction(env, policy_matrix, alpha=0.1, gamma=0.999, tot_epoch=50000):
    """
    Алгоритм пассивного обучения с подкреплением TD(0).

    :param env: среда в которой агент производит действия
    :param policy_matrix: матрица политики действий
    :param gamma: дисконтирующий фактор
    :param tot_epoch: общее число эпох для оценки параметров
    :param print_epoch: число эпох после которых необходимо распечатать текущее состояние оценок ценностей состояний.
    :return: матрицу ценностей действий для состояний (Q)
    """

    # Матрица, содержащая сумму ожидаемых вознаграждений, полученных после первого
    # наблюдения состояния для каждого эпизода.
    utility_matrix = np.zeros((3, 4))

    for epoch in range(tot_epoch):
        # Сбрасываем состояние среды и получаем начальное состояние.
        observation = env.reset(exploring_starts=True)
        for step in range(1000):
            # Выбираем действие для текущего состояния согласно имеющейся политике
            action = policy_matrix[observation[0], observation[1]]
            # Совершаем действие, получаем следующее состояние и вознаграждение.
            new_observation, reward, done = env.step(action)
            # Обновляем матрицу ценностей состояний в соответствии с правилом TD(0).
            utility_matrix = update_utility(utility_matrix, observation, new_observation, reward, alpha, gamma)
            observation = new_observation
            if done:
                break
    return utility_matrix, tot_epoch


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
    utility, tot_epoch = td_prediction(env, policy_matrix)
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility)


if __name__ == "__main__":
    main()