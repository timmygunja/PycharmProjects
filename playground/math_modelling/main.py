import numpy

matrix = numpy.array([
    [0.47, 0, 0, 0, 0, 0.43, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    [0.38, 0.1, 0.27, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
    [0, 0.02, 0.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
    [0, 0, 0.25, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
    [0, 0, 0, 0, 0.55, 0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
    [0, 0.3, 0.2, 0, 0.18, 0.02, 0.18, 0, 0.12, 0, 0, 0, 0, 0, 0],  # 6
    [0.15, 0, 0, 0.1, 0, 0, 0.03, 0.29, 0.15, 0.28, 0, 0, 0, 0, 0],  # 7
    [0, 0, 0, 0, 0, 0.73, 0, 0.08, 0, 0, 0, 0.09, 0, 0, 0.1],  # 8
    [0, 0, 0, 0, 0, 0.21, 0, 0.02, 0.17, 0, 0, 0.26, 0.23, 0, 0.11],  # 9
    [0, 0, 0, 0, 0, 0.27, 0, 0, 0, 0.24, 0.27, 0.22, 0, 0, 0],  # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21, 0.34, 0, 0.27, 0.09, 0.09],  # 11
    [0, 0, 0, 0, 0, 0, 0, 0.34, 0, 0, 0.27, 0.27, 0.12, 0, 0],  # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.48, 0, 0.05, 0.47, 0],  # 13
    [0, 0, 0, 0, 0, 0, 0, 0.17, 0.41, 0, 0, 0, 0, 0.05, 0.37],  # 14
    [0, 0, 0, 0, 0, 0, 0, 0.26, 0.04, 0.1, 0.09, 0, 0, 0.24, 0.27]])  # 15

''' №1 '''
st = 5  # Начальное состояние
fin = 12  # Конечное состояние
k = 6  # Число шагов

res = numpy.linalg.matrix_power(matrix, k)[st - 1][fin - 1]
print('p = ', res)
print()


''' №2 '''
A = numpy.array([0.1, 0.1, 0.14, 0.03, 0.04, 0.05, 0.09, 0.03, 0.08, 0.07, 0.06, 0.07, 0.06, 0.04, 0.04])
k = 7

res = A.dot(numpy.linalg.matrix_power(matrix, k - 1))
print('Ответ:  \n', res)
print()


''' №3 '''
st = 9  # Начальное состояние
fin = 1  # Конечное состояние
k = 10  # Число шагов

matrix_2 = numpy.copy(matrix)
n = range(len(matrix))

for i in range(k - 1):

    matrix_2 = numpy.array([
        [sum(matrix[v1, v3] * matrix_2[v3, v2] if v3 != v2 else 0 for v3 in n) for v2 in n] for v1 in n
    ])

res = matrix_2[st - 1, fin - 1]
print(res)
print()


'''ИСПРАВЛЕННЫЙ ВАРИАНТ'''
matrix_2 = numpy.copy(matrix)
n = range(len(matrix))
matrix_4 = numpy.copy(matrix)


def mult(matrix, matrix_2):
    matrix_3 = numpy.array([
        [sum(matrix[v1, v3] * matrix_2[v3, v2] if v3 != v2 else 0 for v3 in n) for v2 in n] for v1 in n
    ])

    return matrix_3


for i in range(k - 1):
    matrix_4 = mult(matrix, matrix_4)

res = matrix_4[st - 1, fin - 1]
print(res)
print()




''' №4 '''
st = 1  # Начальное состояние
fin = 10  # Конечное состояние
k = 10  # Число шагов

matrix_2, matrix_3 = numpy.copy(matrix), numpy.copy(matrix)

for i in range(k - 1):
    n = range(len(matrix))

    matrix_2 = numpy.array([
        [sum(matrix[v1, v3] * matrix_2[v3, v2] if v3 != v2 else 0 for v3 in n) for v2 in n] for v1 in n
    ])

    matrix_3 += matrix_2

res = matrix_3[st - 1, fin - 1]
print(res)
print()


''' №5 '''
st = 10  # Начальное состояние
fin = 11  # Конечное состояние

matrix_2, matrix_3 = numpy.copy(matrix), numpy.copy(matrix)
n = range(len(matrix))

for i in range(993):

    matrix_2 = numpy.array([
        [sum(matrix[v1, v3] * matrix_2[v3, v2] if v3 != v2 else 0 for v3 in n) for v2 in n] for v1 in n
    ])

    matrix_3 += matrix_2 * i

res = matrix_3[st - 1, fin - 1]
print(res)
print()


'''ИСПРАВЛЕННЫЙ ВАРИАНТ...?'''
st = 10  # Начальное состояние
fin = 11  # Конечное состояние

matrix_2, matrix_3 = numpy.copy(matrix), numpy.copy(matrix)
n = range(len(matrix))

for i in range(993):
    matrix_4 = numpy.array([
        [sum(matrix[v1, v3] * matrix_2[v3, v2] if v3 != v2 else 0 for v3 in n) for v2 in n] for v1 in n
    ])

    matrix_3 += matrix_4 * i

res = matrix_3[st - 1, fin - 1]
print(res)
print()


''' №6 '''
from functools import lru_cache  # Декоратор для сохранения результатов
                                 # последних вызовов функции

st = 15  # Начальное состояние
k = 9  # Число шагов

matrix_2 = numpy.copy(matrix)
p_j = numpy.linalg.matrix_power


@lru_cache()
def f_j(k):
    return p_j(matrix_2, k) - sum([f_j(i) * p_j(matrix_2, k - i) for i in range(1, k)])


res = numpy.diagonal(f_j(k))
print(res[st - 1])
print()


''' №7 '''
st = 2  # Начальное состояние
k = 7  # Число шагов

matrix_2 = numpy.copy(matrix)
p_j = numpy.linalg.matrix_power
lst = []


@lru_cache()
def f_j_2(k):
    mtrx = p_j(matrix_2, k) - sum([f_j_2(i) * p_j(matrix_2, k - i) for i in range(1, k)])
    lst.append(numpy.diagonal(mtrx))
    return mtrx


f_j_2(k)
res = sum(lst)
print(res[st - 1])
print()


''' №8 '''
st = 10  # Начальное состояние

matrix_2 = numpy.copy(matrix)
p_j = numpy.linalg.matrix_power
lst = []


@lru_cache(maxsize=None)
def f_j_3(k, p_j):
    mtrx = p_j(matrix_2, k) - sum([f_j_3(i, p_j) * p_j(matrix_2, k - i) for i in range(1, k)])
    lst.append(numpy.diagonal(mtrx) * k)
    return mtrx


f_j_3(993, p_j)
res = sum(lst)
print(res[st - 1])
print()


''' №9 '''
matrix_final = numpy.copy(matrix).transpose()
numpy.fill_diagonal(matrix_final, numpy.diagonal(matrix_final) - 1)
matrix_final[-1, :] = 1

matrix_4 = numpy.zeros(len(matrix_final))
matrix_4[-1] = 1

# Произведение скаляров
res = numpy.linalg.inv(matrix_final).dot(matrix_4)
print(res)
print()





''' ЗАДАНИЕ 2 '''

''' a) '''
lmbd = 48  # лямбда
m = 4
miu = 19  # мью
n = 5

# Транспонированная матрица
mtrx_t = matrix.T
print(mtrx_t)


mtrx_p = numpy.zeros((m + n + 1, m + n + 1))
# Задаем матрицу переходов
for i in range(m + n):
    mtrx_p[i, i + 1] = lmbd
    if i < m:
        mtrx_p[i + 1, i] = miu * (i + 1)
    else:
        mtrx_p[i + 1, i] = miu * m

print(mtrx_p)
print()


# Матрица с суммами строк по диагонали
lst = []

for i in range(mtrx_p.shape[0]):
    lst.append(mtrx_p[i, :].sum())

mtrx_d = numpy.diag(lst)
print(mtrx_d)
print()

mtrx_m = mtrx_p - mtrx_d
print(mtrx_m)
print()

from copy import deepcopy

M_ = deepcopy(mtrx_m)
M_[-1, :] = 1
print(M_)
print()

mtrx_b = numpy.zeros(M_.shape[0])
mtrx_b[-1] = 1
print(mtrx_b)
print()


X = numpy.linalg.inv(M_).dot(mtrx_b)
sumX = 0

for i in X:
    print(i)
    sumX += i

# Проверка
print()
print("Сумма всех вероятностей:", sumX)


''' b) '''
print(X[-1])


''' c) '''
print("q =", 1 - X[-1])
print("A =", (1 - X[-1]) * lmbd)
print()


''' d) '''
lst = []

for i in range(1, n + 1):
    lst.append(i * X[m + i])  # i * p_m+i

res = sum(lst)
print(res)
print()


''' e) '''
lst = []

for i in range(n):
    lst.append((i + 1) / (m * miu) * X[m + i])  # (i + 1) / (m * miu) * p_m+i

res = sum(lst)
print(res)
print()


''' f) '''
lst = []

for i in range(1, m + 1):
    lst.append(i * X[i])  # i * p_i

sum_left = sum(lst)

lst = []

for i in range(m + 1, m + n + 1):
    lst.append(m * X[i])  # m * p_i

sum_right = sum(lst)

res = sum_left + sum_right
print(res)
print()


''' g) '''
print(sum(X[:m]))


''' h) '''
print(1 / lmbd)
print()
