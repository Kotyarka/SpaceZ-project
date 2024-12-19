import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

m0 = 287000  # Начальная масса ракеты
k = 292  # Скорость расхода топлива (кг/с)
Cf = 0.2  # Коэффициент сопротивления
S = 10.3  # Площадь поперечного сечения ракеты
p0 = 101325  # Давление на уровне моря (Па)
M = 0.02896  # Молярная масса воздуха (кг/моль)
F_thrust = 4941000 # Мощность двигателя
R = 8.314  # Универсальная газовая постоянная (Дж/(моль·К))
T = 293  # Температура воздуха (К)
b = 0.1 # Коэффициент изменения угла наклона
phi0 = 0 # Начальный угол
g = 9.81  # Ускорение свободного падения (м/с^2)
G = 6.67430e-11  # Гравитационная постоянная (м^3/(кг·с^2))
M_earth = 5.972e24  # Масса Земли (кг)в байтах
R_earth = 6.371e6  # Радиус Земли (м)
s1 = False
s2 = False


# Функция для расчета плотности воздуха
def rho(y):
    return (p0 * M) / (R * T) * np.exp(-(M * g) / (R * T) * y)

# Система дифференциальных уравнений
def equations(t, y):
    global m0, s1, s2, F_thrust, s3
    if t > 120 and s1 == False:
        m0 = 244000
        F_thrust = 941000
        s1 = True

    vx, vy, h = y
    m = m0 - k * t
    phi = phi0 + b * t
    
    F_gravity = G * m * M_earth / (R_earth + h) ** 2
    F_drag = 0.5 * Cf * S * (vx ** 2 + vy ** 2) * rho(h)

    dvx_dt = (F_thrust - F_drag) / m
    dvy_dt = (F_thrust - F_drag - F_gravity) / m
    dh_dt = vy

    return [dvx_dt, dvy_dt, dh_dt, ]


# Начальные условия
y0 = [0, 0, 82]  # Начальные значения vx, vy, h


# Временной интервал
t_span = (0, 260)  # От 0 до 260 секунд
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Решение системы ОДУ
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Вывод результатов
table = pd.read_excel("Данные о полете (исследуемые).xls")
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[0], label='math vy(t)')
x = table.values[:, 0]
y = table.values[:, 4]
plt.plot(x, y, label='ksp vy(t)')
plt.xlabel('Время (с)')
plt.ylabel('Скорость  (м/с)')
plt.legend()

plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[2], label='math h(t)')
x = table.values[:, 0]
y = table.values[:, 2]
plt.plot(x, y, label='ksp h(t)')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.legend()

plt.tight_layout()
plt.show()
