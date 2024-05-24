import numpy as np
from tqdm.notebook import tqdm

def viterby(
        observations: list,
        init_prob: list,
        transition_prob: list[list],
        emision_prob: list[dict]
        ):
    """ Имплементация алгоритма Витерби для поиска наиболее
    вероятного пути

    Args:
        observations (list):
        Наблюдения

        init_prob (list):
        Вероятности начальныйх состояний длиной len(init_prob)

        transition_prob (list[list]):
        Матрица перезодов размера len(init_prob)*len(init_prob)

        emision_prob (list[dict]):
        Матрица эмисий размера len(init_prob)*len(set(observations))
    """
    n_st = len(init_prob)  # кол-во состояний
    n_ob = len(observations)  # кол-во наблюдений
    grid = np.zeros(shape=(n_st, n_ob))
    pointers = np.zeros(shape=(n_st, n_ob))
    for s in range(n_st):
        grid[s, 0] = init_prob[s]*emision_prob[s][observations[0]]

    for o in range(1, n_ob):
        for s in range(n_st):
            values = [grid[k, o-1]*transition_prob[k][s]*emision_prob[k][observations[o]] for k in range(n_st)]
            k_max = np.argmax(np.array(values))
            grid[s, o] = grid[k_max, o-1]*transition_prob[k_max][s]*emision_prob[s][observations[o]]
            pointers[s, o] = k_max
    best_path = []
    k = np.argmax(grid[:, -1])
    for o in range(n_ob-1, -1, -1):
        best_path.insert(0, k)
        k = int(pointers[k, o])
    return best_path


def forward(
        observations: list,
        init_prob: list,
        transition_prob: list[list],
        emision_prob: list[dict]
        ):
    """ Алгоритм прямого прохода для расчета P(X)

    Args:
        observations (list):
        Наблюдения

        init_prob (list):
        Вероятности начальныйх состояний длиной len(init_prob)

        transition_prob (list[list]):
        Матрица перезодов размера len(init_prob)*len(init_prob)

        emision_prob (list[dict]):
        Матрица эмисий размера len(init_prob)*len(set(observations))
    """
    grid = _forward(
        observations,
        init_prob,
        transition_prob,
        emision_prob
        )
    return sum(grid[:, -1])


def posterior_decoding(
        observations: list,
        init_prob: list,
        transition_prob: list[list],
        emision_prob: list[dict],
        inference_mode: bool = False
        ):
    """ Вычисление апостериорных вероятностей P(pi_i = k | X), при помощи
    forward и backward прохода

    Args:
        observations (list):
        Наблюдения

        init_prob (list):
        Вероятности начальныйх состояний длиной len(init_prob)

        transition_prob (list[list]):
        Матрица перезодов размера len(init_prob)*len(init_prob)

        emision_prob (list[dict]):
        Матрица эмисий размера len(init_prob)*len(set(observations))

        inference_mode (bool):
        Если False, то возвращает вероятность P(state_t = 0) в каждый момент времени t,
        иначе возвращает самую вероятную последовательность состояний.
    """
    if inference_mode:
        # TODO
        return

    # Forward
    f_grid = _forward(
        observations,
        init_prob,
        transition_prob,
        emision_prob
        )
    # Backward
    b_grid = _backward(
        observations,
        init_prob,
        transition_prob,
        emision_prob
        )
        
    result = []
    p = sum(f_grid[:, -1])  # P(X)
    for t in range(len(observations)):
        f_i = f_grid[0, t]
        b_i = b_grid[0, t]
        posterior = f_i*b_i/p  # P(pi_i = 0|X) = f_i*b_i/P(X)
        result.append(posterior)
    return result


def baum_welch(
        observations: list,  
        A: list[list] = None,
        B: list[dict] = None,
        init_prob: list = None,
        n_st: int = 2,
        n_sym: int = 6,
        n_iter: int = 100):
    """Имплементаций алгоритма Баума-Велша для оценки параметров
    скрытой Марковской цепи

    Args:
        observations (list): 
        Наблюдения.

        A (list[list], optional): 
        Начальное состояние матрицы переходов. Defaults to None.

        B (list[dict], optional): 
        Начальное состояние матрицы эмиссий. Defaults to None.
        init_prob (list, optional): 
        
        Начальное состояний матрицы начальных состояний ಠ__ಠ. Defaults to None.
        n_st (int, optional): 
        Кол-во состояний. Defaults to 2.

        n_sym (int, optional): 
        Кол-во симоволов. Defaults to 6.

        n_iter (int, optional): 
        Кол-во итераций. Defaults to 100.
    """    

    if A is None:
        A = np.ones((n_st, n_st))
        A = A / np.sum(A, axis=1)

    if B is None:
        B = np.ones((n_st, n_sym))
        B = B / np.sum(B, axis=1).reshape((-1, 1))

    if init_prob is None:
        init_prob = np.ones(n_st)/n_st

    n_ob = len(observations)
    for _ in tqdm(range(n_iter)):
        # Forward
        f_grid = _forward(
            observations,
            init_prob,
            A,
            B).T
        # Backward
        b_grid = _backward(
            observations,
            init_prob,
            A,
            B).T

        grid = np.zeros((n_st, n_st, n_ob - 1))
        for t in range(1, n_ob):
            p_x = np.dot(np.dot(f_grid[t-1, :].T, A) * B[:, observations[t]].T, b_grid[t, :])  
            for i in range(n_st):
                i_sum = f_grid[t-1, i] * A[i, :] * B[:, observations[t]].T * b_grid[t, :].T  
                grid[i, :, t-1] = i_sum / p_x

        s_grid = np.sum(grid, axis=1)  
        A = np.sum(grid, 2) / np.sum(s_grid, axis=1).reshape((-1, 1))

        s_grid = np.hstack((s_grid, np.sum(grid[:, :, n_ob - 2], axis=0).reshape((-1, 1))))

        p_x = np.sum(s_grid, axis=1)
        for i in range(n_sym):
            B[:, i] = np.sum(s_grid[:, observations == i], axis=1)

        B = np.divide(B, p_x.reshape((-1, 1)))

    return {"Transition": A, "Emission": B}


def _forward(
        observations: list,
        init_prob: list,
        transition_prob: list[list],
        emision_prob: list[dict]
        ):
    """ Подручный алгоритм для реализации прямого прохода

    Args:
        observations (list):
        Наблюдения

        init_prob (list):
        Вероятности начальныйх состояний длиной len(init_prob)

        transition_prob (list[list]):
        Матрица перезодов размера len(init_prob)*len(init_prob)

        emision_prob (list[dict]):
        Матрица эмисий размера len(init_prob)*len(set(observations))
    """
    n_st = len(init_prob)  # кол-во состояний
    n_ob = len(observations)  # кол-во наблюдений
    grid = np.zeros(shape=(n_st, n_ob))
    for s in range(n_st):
        grid[s, 0] = init_prob[s]*emision_prob[s][observations[0]]

    for o in range(1, n_ob):
        for s in range(n_st):
            state = sum([grid[k][o-1]*transition_prob[k][s] for k in range(n_st)])
            symbol = emision_prob[s][observations[o]]
            grid[s, o] = symbol*state
    return grid


def _backward(
        observations: list,
        init_prob: list,
        transition_prob: list[list],
        emision_prob: list[dict]
        ):
    """ Подручный алгоритм для реализации обратого прохода

    Args:
        observations (list):
        Наблюдения

        init_prob (list):
        Вероятности начальныйх состояний длиной len(init_prob)

        transition_prob (list[list]):
        Матрица перезодов размера len(init_prob)*len(init_prob)

        emision_prob (list[dict]):
        Матрица эмисий размера len(init_prob)*len(set(observations))
    """
    n_st = len(init_prob)  # кол-во состояний
    n_ob = len(observations)  # кол-во наблюдений
    grid = np.zeros(shape=(n_st, n_ob+1))
    for s in range(n_st):
        grid[s, -1] = 1

    for o in range(n_ob-1, -1, -1):
        for s in range(n_st):
            value = sum([grid[k][o+1]*transition_prob[s][k]*emision_prob[k][observations[o]] for k in range(n_st)])
            grid[s, o] = value
    return grid[:, :-1]
