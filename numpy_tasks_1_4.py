"""Лабораторная работа: решения 1.1-4.4; запуск python numpy_tasks_1_4.py."""
import timeit

print('========== ЗАДАЧА 1 ==========')
import numpy as np
from decimal import Decimal

arr32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
sum32 = np.sum(arr32, dtype=np.float32)
exact = Decimal('0.1') + Decimal('0.2') + Decimal('0.3')
err = Decimal.from_float(float(sum32)) - exact

print('1.1. Сумма float32:', format(float(sum32), '.17g'))
print('     Точное десятичное значение:', exact)
print('     Абсолютная ошибка:', abs(err))

print('\n1.2. Машинное эпсилон:')
print('     float32:', np.finfo(np.float32).eps)
print('     float64:', np.finfo(np.float64).eps)

a32, b32 = np.float32(1.00000001), np.float32(1.0)
a64, b64 = np.float64(1.00000001), np.float64(1.0)
print('\n1.3. Ожидаемая разность: 0.00000001')
print('     float32:', a32 - b32)
print('     float64:', a64 - b64)
print('     float32 округляет 1.00000001 до 1.0, поэтому разность равна 0.')

print("\n========== ЗАДАЧА 2 ==========")
import numpy as np

# Отрицательные числа оставляем специально для задания 2.3.
A = np.arange(-10, 10).reshape(4, 5)
print('Исходный A (4 x 5):\n', A)

print('\n2.1. Вторая строка (индекс 1):', A[1, :])
print('     Столбец с индексом 3:', A[:, 3])
print('     Строки с индексами 1,2,3 и столбцы 2,3,4:\n', A[1:4, 2:5])

# np.ix_ делает декартово произведение индексов: результат 3 x 2.
rows = [0, 2, 3]
cols = [1, 4]
selected = A[np.ix_(rows, cols)]
print('\n2.2. Fancy-индексация (3 x 2):\n', selected)

nonnegative = A.copy()
nonnegative[nonnegative < 0] = 0
print('\n2.3. Отрицательные заменены нулями:\n', nonnegative)

B = np.arange(1, 5).reshape(4, 1)
C = np.array([[10, 20, 30, 40, 50]])
D = B + C
print('\n2.4. B.shape:', B.shape, 'C.shape:', C.shape)
print('     Broadcasting: B + C, форма', D.shape, '\n', D)
assert D.shape == (4, 5)

print("\n========== ЗАДАЧА 3 ==========")
import numpy as np
from pathlib import Path

file_path = Path('numpy_memmap.dat')
rng = np.random.default_rng(42)

# 3.1. Создание массива на диске, ~8 МБ.
mm = np.memmap(file_path, dtype='float64', mode='w+', shape=(1000, 1000))
for start in range(0, 1000, 100):
    mm[start:start + 100, :] = rng.random((100, 1000))
mm.flush()
del mm

# 3.2. Открываем только отображение файла, не загружая все 8 МБ в память.
mapped = np.memmap(file_path, dtype='float64', mode='r', shape=(1000, 1000))
print('3.1. Создан файл:', file_path, '| размер:', file_path.stat().st_size, 'байт')
print('3.2. Первые 10 элементов строки 500:', mapped[500, :10])
print('     Первые 10 элементов столбца 500:', mapped[:10, 500])

def sum_by_rows(m):
    s = 0.0
    for i in range(m.shape[0]):
        s += np.sum(m[i, :])
    return s

def sum_by_columns(m):
    s = 0.0
    for j in range(m.shape[1]):
        s += np.sum(m[:, j])
    return s

assert np.isclose(sum_by_rows(mapped), sum_by_columns(mapped))
print('3.3. Суммы при обоих способах обхода совпадают (с допустимой погрешностью).')
t_row = min(timeit.repeat(lambda: sum_by_rows(mapped), repeat=3, number=3)) / 3
t_col = min(timeit.repeat(lambda: sum_by_columns(mapped), repeat=3, number=3)) / 3
print(f'     row-major: {t_row:.6f} сек; column-major: {t_col:.6f} сек')
print('\n========== ЗАДАЧА 4 ==========')
import numpy as np
from scipy import sparse

# 4.1. Размер 5x5, плотность 0.2 => 5 ненулевых элементов.
csr = sparse.random(5, 5, density=0.2, format='csr', random_state=42)
print('4.1. CSR-матрица:\n', csr.toarray())
print('     Ненулевых элементов:', csr.nnz, 'из 25; плотность:', csr.nnz / 25)

# 4.2. Переводим в другие форматы хранения.
coo = csr.tocoo()
csc = csr.tocsc()
print('\n4.2. Форматы:', type(csr).__name__, type(coo).__name__, type(csc).__name__)
print('     COO (строка, столбец, значение):')
for r, c, value in zip(coo.row, coo.col, coo.data):
    print(f'     ({r}, {c}) = {value:.6f}')
print('     CSC.indptr:', csc.indptr, 'CSC.indices:', csc.indices)

# 4.3. Матрично-векторное произведение.
v = np.arange(1, 6, dtype=float)
result = csr @ v
print('\n4.3. Вектор:', v)
print('     CSR @ вектор:', result)
np.testing.assert_allclose(result, csr.toarray() @ v)
# 4.4. Сравнение на одинаковой матрице и одинаковом векторе.
big_sparse = sparse.random(1000, 1000, density=0.01, format='csr', random_state=123)
big_dense = big_sparse.toarray()
vector = np.random.default_rng(123).random(1000)
np.testing.assert_allclose(big_sparse @ vector, big_dense @ vector, rtol=1e-12, atol=1e-12)

print('4.4. 1000x1000, density=0.01')
print('     CSR хранение (data, indices, indptr):',
      big_sparse.data.nbytes + big_sparse.indices.nbytes + big_sparse.indptr.nbytes, 'байт')
print('     Dense хранение:', big_dense.nbytes, 'байт')
print('     Результаты умножения совпадают.')
t_sparse = min(timeit.repeat(lambda: big_sparse @ vector, repeat=3, number=10)) / 10
t_dense = min(timeit.repeat(lambda: big_dense @ vector, repeat=3, number=10)) / 10
print(f'     sparse @ vector: {t_sparse:.8f} сек')
print(f'     dense  @ vector: {t_dense:.8f} сек')
