"""
Дана функция calculate(x).
Напишите программу, которая создает пул из 5 процессов и распределяет в этом пуле вычисление
функции на промежутке х от 0 до 1 с шагом 0,1.
"""
import sys
from multiprocessing import Pool


def calculate(x):
    return x * x


def main():
    # Создаем пул
    pool = Pool(processes=5)
    # Создаем список аргументов x для функции calculate
    data = [round(i * 0.1, 2) for i in range(11)]

    # Получаем результат выполнения
    res = list(pool.map(calculate, data))

    # Вывод на экран и запись в файл
    with open(sys.argv[1], 'a') as f:
        for arg, ans in zip(data, res):
            print(f"x = {arg}, результат = {ans}")
            f.write(f"x = {arg}, результат = {ans}\n")


if __name__ == "__main__":
    main()

