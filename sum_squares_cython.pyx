# cython: language_level=3
# Реализация для списка целых чисел.
# Важное условие: сумма квадратов должна помещаться в signed long long.

cpdef long long sum_squares_cython(list values):
    cdef Py_ssize_t i, n = len(values)
    cdef long long x, total = 0
    for i in range(n):
        x = <long long>values[i]
        total += x * x
    return total
