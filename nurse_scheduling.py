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

# Global model parameters: problem size
n_nurses = 3      # number of nurses
n_days = 11       # number of days
size = n_days * n_nurses

# Parameters for the hard constraint (no consecutive working days)
a = 3.5

# Parameters for the hard shift constraint (at least one nurse working per day)
lagrange_hard_shift = 1.3
workforce = 1
effort = 1

# Parameters for the soft constraint (even workload distribution)
lagrange_soft_nurse = 0.3
preference = 1
min_duty_days = int(n_days / n_nurses)

# Function to obtain a 1D index from (nurse index, day index)
def get_index(nurse_index, day_index):
    return nurse_index * n_days + day_index

# Inverse function: get (nurse, day) from a 1D index
def get_nurse_and_day(index):
    nurse_index, day_index = divmod(index, n_days)
    return nurse_index, day_index

print("\nBuilding binary quadratic model...")

# Hard constraint: a nurse should not work on two consecutive days
J = defaultdict(int)
for nurse in range(n_nurses):
    for day in range(n_days - 1):
        nurse_day_1 = get_index(nurse, day)
        nurse_day_2 = get_index(nurse, day + 1)
        J[nurse_day_1, nurse_day_2] = a

# Build the Q-matrix (QUBO) based on matrix J
Q = deepcopy(J)

# Hard shift constraint: at least one nurse works each day
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

# Soft constraint: even distribution of workdays among nurses
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

# Energy offset
e_offset = (lagrange_hard_shift * n_days * workforce ** 2) + (lagrange_soft_nurse * n_nurses * min_duty_days ** 2)
bqm = BinaryQuadraticModel.from_qubo(Q, offset=e_offset)

print("\nSending problem to QDeepHybridSolver...")

# Convert the BQM to a QUBO dictionary and offset
qubo, offset = bqm.to_qubo()

# Convert the QUBO dictionary to a numpy array
matrix = np.zeros((size, size))
for (i, j), value in qubo.items():
    matrix[i, j] = value

# Initialize the QDeepHybridSolver
solver = QDeepHybridSolver()
solver.token = "your-auth-token-here"

# Solve the problem by passing the numpy array
results = solver.solve(matrix)

# Extract the solution (assumes the result returns a dictionary with key 'sample')
smpl = results['sample']

print("\nBuilding schedule and checking constraints...\n")
# Build the schedule based on the obtained solution
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

# Plot the schedule
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

# Print the schedule to the console
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
