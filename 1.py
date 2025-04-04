import numpy as np
import matplotlib.pyplot as plt

def calculate_motion(a_func, t_max, v0=0, x0=0, n_steps=1000):
    """
    Численное интегрирование уравнений движения
    """
    t = np.linspace(0, t_max, n_steps)
    dt = t[1] - t[0]
    
    a = np.zeros_like(t)
    v = np.zeros_like(t)
    x = np.zeros_like(t)
    
    v[0] = v0
    x[0] = x0
    
    for i in range(1, n_steps):
        a[i] = a_func(t[i])
        v[i] = v[i-1] + a[i] * dt
        x[i] = x[i-1] + v[i] * dt
    
    return t, x, v, a

def plot_results(t, x, v, a, title):
    """Визуализация результатов"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, a, 'r')
    plt.title(f'Ускорение: {title}')
    plt.ylabel('a(t) [м/с²]')
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, v, 'b')
    plt.title(f'Скорость: {title}')
    plt.ylabel('v(t) [м/с]')
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, x, 'g')
    plt.title(f'Координата: {title}')
    plt.xlabel('Время t [с]')
    plt.ylabel('x(t) [м]')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

# 1. Линейное ускорение
def a1(t):
    return 6*t + 2

# 2. Квадратичное ускорение
def a2(t):
    return t**2

# 3. Кусочно-заданное ускорение (из классной работы)
def a3(t):
    if t < 2:
        return 0.5*t  # линейный рост
    elif 2 <= t < 5:
        return 1.0    # постоянное ускорение
    else:
        return 1.0 - 0.2*(t-5)  # линейный спад

# 4. Синусоидальное ускорение
def a4(t):
    return np.sin(t)

# 5. Экспоненциальные ускорения
def a5_exp(t):
    return np.exp(t)

def a5_exp_comb(t):
    return np.exp(-t) - 2*np.exp(-2*t)

# 6. Затухающие колебания
def a6(t):
    return 10*np.exp(-t)*np.cos(8*t)

# Проверка для всех случаев
cases = [
    ("Линейное a(t)=6t+2", a1, 5, 0, 0),
    ("Квадратичное a(t)=t²", a2, 4, 0, 0),
    ("Кусочно-заданное (классная работа)", a3, 8, 0, 0),
    ("Синусоидальное a(t)=sin(t)", a4, 4*np.pi, 0, 0),
    ("Экспоненциальное a(t)=e^t", a5_exp, 3, 0, 0),
    ("Комбинация экспонент a(t)=e^{-t}-2e^{-2t}", a5_exp_comb, 5, 0, 0
