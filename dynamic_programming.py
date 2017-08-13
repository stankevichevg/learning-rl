import numpy as np

from utils import print_policy


def return_state_utility(v, T, u, reward, gamma):
    """
    Возвращает ценность состояния.

    @param v векор длиной равной числу состояний с единицей для заданного состояния и нулями в остальных позициях
    @param T матрица вероятностей переходов
    @param u вектор ценности состояний
    @param reward вознаграждение для данного состояния
    @param gamma дисконтирующий множетель
    @return ценность данного состояния
    """
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    # в соответствии с уравнением Белмана, ценность состояния это его вознаграждение
    # в сумме с дисконтированным ожидаемым вознаграждением при оптимальном действии (действие для которого
    # ожидаемое вознаграждение максимально)
    return reward + gamma * np.max(action_array)


def print_result(u, iteration, delta, gamma, epsilon):
    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")


def return_policy_evaluation(p, u, r, T, gamma):
    """
    Возвращает вектор ценности состояний, при условии следовании заданной политике.

    @param p вектор политики, определяет какое действие совершать в каждом состоянии
    @param u вектор ценности состояний
    @param r вознаграждение для данного состояния
    @param T матрица вероятностей переходов
    @param gamma дисконтирующий множетель
    @return вектор ценности состояний, при условии следовании заданной политике
    """
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1,12))
            v[0,s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return u


def return_expected_action(u, T, v):
    """
    Возвращает оптимальное с точки зрения ожидаемого вознаграждения действие.

    @param u вектор ценности состояний
    @param T матрица вероятностей переходов
    @param v вектор вероятностей начального состояния
    @return действие (int)
    """
    actions_array = np.zeros(4)
    for action in range(4):
        # ожидаемая ценность от действия a в состоянии s, в соответствии с T и u.
        actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)


def value_iteration(T, r, tot_states=12, gamma=0.999, epsilon=0.01):
    iteration = 0
    # начальное состояние вектора ценностей состояний
    u = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0]
    )
    while True:
        delta = 0.0
        u1 = u.copy()
        iteration += 1
        for s in range(tot_states):
            reward = r[s]
            v = np.zeros((1,tot_states))
            v[0,s] = 1.0
            u[s] = return_state_utility(v, T, u1, reward, gamma)
            delta = max(delta, np.abs(u[s] - u1[s]))
        # если ни для одного состояния не произошло существенного изменения ценности, то мы нашли решение
        if delta < epsilon * (1 - gamma) / gamma:
            print_result(u, iteration, delta, gamma, epsilon)
            break
    return u


def policy_iteration(T, r, tot_states=12, gamma=0.999, epsilon=0.01):
    iteration = 0
    # инициализируем случайную политику
    p = np.random.randint(0, 4, size=tot_states).astype(np.float32)
    # в одно состояние мы не можем попасть
    p[5] = np.NaN
    # в терминальных состояниях ничего не делаем
    p[3] = p[7] = -1
    # начальное состояние вектора ценностей состояний
    u = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0]
    )
    while True:
        iteration += 1
        # 1 - Получаем оценки ценностей состояния для текущей политики
        u_0 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        # Проверяем, может нам пора остановиться
        delta = np.absolute(u - u_0).max()
        if delta < epsilon * (1 - gamma) / gamma:
            break
        for s in range(tot_states):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1, tot_states))
                v[0,s] = 1.0
                # 2 - для текущих оценок ценностей состояний получаем оптимальное действие и обновляем политику
                p[s] = return_expected_action(u, T, v)
        print_policy(p, shape=(3,4))

    print_result(u, iteration, delta, gamma, epsilon)
    return p, u


def main():

    # Матрица вероятностей переходов s -> s' при совержении действия a. p(s, s', a) = T[s, s', a]
    T = np.load("T.npy")

    # Вектор вознаграждений получаемых агентов в каждом состоянии
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    # запускаем алгоритм Value Iteration и находим вектор ценности состояний
    u = value_iteration(T, r)
    # запускаем алгоритм Policy Iteration и находим вектор ценности состояний
    p, u = policy_iteration(T, r)
    print_policy(p, shape=(3,4))


if __name__ == "__main__":
    main()