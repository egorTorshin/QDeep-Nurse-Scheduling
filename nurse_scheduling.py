from qdeepsdk import QDeepHybridSolver
import numpy as np
from dimod import BinaryQuadraticModel
from collections import defaultdict
from copy import deepcopy
import matplotlib

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

# Общие параметры модели: размер задачи
n_nurses = 3      # число медсестер
n_days = 11       # число дней
size = n_days * n_nurses

# Параметры для жёсткого ограничения (работа в соседние дни)
a = 3.5

# Параметры для ограничения "хотя бы одна медсестра в день"
lagrange_hard_shift = 1.3
workforce = 1
effort = 1

# Параметры для мягкого ограничения (выравнивание нагрузки)
lagrange_soft_nurse = 0.3
preference = 1
min_duty_days = int(n_days / n_nurses)

# Функция для получения 1D индекса по (индекс медсестры, индекс дня)
def get_index(nurse_index, day_index):
    return nurse_index * n_days + day_index

# Обратная функция: из 1D индекса получаем (медсестра, день)
def get_nurse_and_day(index):
    nurse_index, day_index = divmod(index, n_days)
    return nurse_index, day_index

print("\nBuilding binary quadratic model...")

# Ограничение: одна медсестра не должна работать в два подряд идущих дня
J = defaultdict(int)
for nurse in range(n_nurses):
    for day in range(n_days - 1):
        nurse_day_1 = get_index(nurse, day)
        nurse_day_2 = get_index(nurse, day + 1)
        J[nurse_day_1, nurse_day_2] = a

# Формируем Q-матрицу (QUBO) на основе матрицы J
Q = deepcopy(J)

# Ограничение: хотя бы одна медсестра работает каждый день
for nurse in range(n_nurses):
    for day in range(n_days):
        ind = get_index(nurse, day)
        Q[ind, ind] += lagrange_hard_shift * (effort ** 2 - 2 * workforce * effort)

for day in range(n_days):
    for nurse1 in range(n_nurses):
        for nurse2 in range(nurse1 + 1, n_nurses):
            ind1 = get_index(nurse1, day)
            ind2 = get_index(nurse2, day)
            Q[ind1, ind2] += 2 * lagrange_hard_shift * effort ** 2

# Ограничение: равномерное распределение рабочих дней (мягкое ограничение)
for nurse in range(n_nurses):
    for day in range(n_days):
        ind = get_index(nurse, day)
        Q[ind, ind] += lagrange_soft_nurse * (preference ** 2 - 2 * min_duty_days * preference)

for nurse in range(n_nurses):
    for day1 in range(n_days):
        for day2 in range(day1 + 1, n_days):
            ind1 = get_index(nurse, day1)
            ind2 = get_index(nurse, day2)
            Q[ind1, ind2] += 2 * lagrange_soft_nurse * preference ** 2

# Смещение энергии
e_offset = (lagrange_hard_shift * n_days * workforce ** 2) + (lagrange_soft_nurse * n_nurses * min_duty_days ** 2)
bqm = BinaryQuadraticModel.from_qubo(Q, offset=e_offset)

print("\nSending problem to QDeepHybridSolver...")

# Преобразуем BQM в QUBO-словарь и смещение
qubo, offset = bqm.to_qubo()

# Преобразование QUBO-словаря в numpy-массив
matrix = np.zeros((size, size))
for (i, j), value in qubo.items():
    matrix[i, j] = value

# Инициализация решателя QDeepHybridSolver
solver = QDeepHybridSolver()
solver.token = "your-auth-token-here"

# Решаем задачу, передавая numpy-массив
results = solver.solve(matrix)

# Извлекаем решение (предполагается, что возвращается словарь с ключом 'sample')
smpl = results['sample']

print("\nBuilding schedule and checking constraints...\n")
# Формируем расписание на основе полученного решения
sched = [get_nurse_and_day(j) for j in range(size) if smpl.get(j, 0) == 1]

def check_hard_shift_constraint(sched, n_days):
    satisfied = [False] * n_days
    for _, day in sched:
        satisfied[day] = True
    return "Satisfied" if all(satisfied) else "Unsatisfied"

def check_hard_nurse_constraint(sched, n_nurses):
    satisfied = [True] * n_nurses
    for nurse, day in sched:
        if ((nurse, day + 1) in sched) or ((nurse, day - 1) in sched):
            satisfied[nurse] = False
    return "Satisfied" if all(satisfied) else "Unsatisfied"

def check_soft_nurse_constraint(sched, n_nurses):
    num_shifts = [0] * n_nurses
    for nurse, _ in sched:
        num_shifts[nurse] += 1
    return "Satisfied" if num_shifts.count(num_shifts[0]) == len(num_shifts) else "Unsatisfied"

print("\tHard shift constraint:", check_hard_shift_constraint(sched, n_days))
print("\tHard nurse constraint:", check_hard_nurse_constraint(sched, n_nurses))
print("\tSoft nurse constraint:", check_soft_nurse_constraint(sched, n_nurses))

# Построение графика расписания
x, y = zip(*sched) if sched else ([], [])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y, x)
width = 1
height = 1
for a_y, a_x in sched:
    if a_y == 0:
        ax.add_patch(Rectangle(xy=(a_x - width/2, a_y - height/2), width=width, height=height,
                               linewidth=1, color='blue', fill=True))
    elif a_y == 1:
        ax.add_patch(Rectangle(xy=(a_x - width/2, a_y - height/2), width=width, height=height,
                               linewidth=1, color='red', fill=True))
    else:
        ax.add_patch(Rectangle(xy=(a_x - width/2, a_y - height/2), width=width, height=height,
                               linewidth=1, color='green', fill=True))
ax.axis('equal')
ax.set_xticks(range(n_days))
ax.set_yticks(range(n_nurses))
ax.set_xlabel("Shifts")
ax.set_ylabel("Nurses")
plt.savefig("schedule.png")

# Вывод расписания в консоль
print("\nSchedule:\n")
for n in range(n_nurses - 1, -1, -1):
    str_row = ""
    for d in range(n_days):
        outcome = "X" if (n, d) in sched else " "
        if d > 9:
            outcome += " "
        str_row += "  " + outcome
    print("Nurse", n, str_row)

str_header_for_output = " " * 11 + "  ".join(map(str, range(n_days)))
print(str_header_for_output, "\n")
print("Schedule saved as schedule.png.")
