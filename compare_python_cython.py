"""Задача 5: запуск после сборки `python setup.py build_ext --inplace`."""
import timeit
from sum_squares_cython import sum_squares_cython

def sum_squares_python(values):
    total = 0
    for x in values:
        total += x * x
    return total

values = [i % 10 for i in range(10**7)]
assert sum_squares_python(values) == sum_squares_cython(values)
python_time = min(timeit.repeat(lambda: sum_squares_python(values), repeat=3, number=1))
cython_time = min(timeit.repeat(lambda: sum_squares_cython(values), repeat=3, number=3)) / 3
print('Число элементов:', len(values))
print('Сумма квадратов:', sum_squares_python(values))
print(f'Python: {python_time:.6f} с')
print(f'Cython: {cython_time:.6f} с')
print(f'Ускорение: {python_time / cython_time:.2f} раза')
