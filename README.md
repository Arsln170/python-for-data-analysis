# Python для анализа данных

- [Лабораторная работа 1. NumPy](lab1.ipynb)
- [Лабораторная работа 2. NumPy и Pandas](lab2.ipynb)

## Запуск

```bash
python -m pip install -r requirements.txt
python -m notebook
```

Задачи 1–4 из первой работы можно запустить отдельно.

```bash
python numpy_tasks_1_4.py
```

Для сравнения Python и Cython в задаче 5:

```bash
python setup.py build_ext --inplace
python compare_python_cython.py
```
