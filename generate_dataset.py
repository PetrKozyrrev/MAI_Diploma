import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.linalg import solve_banded

# Настраиваемые константы генерации
GRID_X = 400 
GRID_Y = 400
VIDEO_SECONDS = 6
FPS = 30
NUM_SOURCES = 1
SOURCES_MOVE = [True]
ALPHA = 300
SOURCE_POWER = 300
DX = DY = 1.0
DT = 0.25

FRAMES = VIDEO_SECONDS * FPS
T = np.zeros((GRID_X, GRID_Y))


class HeatSource:
    def __init__(
        self,
        x,
        y,
        power,
        radius=5.0,
        motion_type='linear',
        vx=0.0,
        vy=0.0,
        cx=None,
        cy=None,
        orbit_radius=50.0,
        a=50.0,
        b=30.0,
        omega=0.05,
    ):
        self.x = x
        self.y = y
        self.x0, self.y0 = x, y
        self.power = power
        self.radius = radius
        self.motion_type = motion_type

        # Параметры кругового движения
        self.cx = cx if cx is not None else x
        self.cy = cy if cy is not None else y
        self.orbit_radius = orbit_radius
        self.omega = omega
        self.angle = 0.0

        # Параметры движения
        self.a = a
        self.b = b
        self.omega = omega
        self.t = 0.0

        if motion_type == 'linear':
            self.vx = vx
            self.vy = vy
        if motion_type == 'chaotic':
            self.vx = np.random.uniform(-0.5, 0.5)
            self.vy = np.random.uniform(-0.5, 0.5)

    def move(self):
        if not SOURCES_MOVE:
            return

        self.t += self.omega

        if self.motion_type == 'lemniscate':
            denom = 1 + np.sin(self.t) ** 2
            self.x = self.x0 + (self.a * np.cos(self.t)) / denom
            self.y = (
                self.y0
                + (self.a * np.sin(self.t) * np.cos(self.t)) / denom
            )

        elif self.motion_type == 'ellipse':
            self.x = self.x0 + self.a * np.cos(self.t)
            self.y = self.y0 + self.b * np.sin(self.t)

        elif self.motion_type == 'chaotic':
            self.vx += np.random.uniform(-0.1, 0.1)
            self.vy += np.random.uniform(-0.1, 0.1)

            self.vx = np.clip(self.vx, -1.5, 1.5)
            self.vy = np.clip(self.vy, -1.5, 1.5)

            self.x += self.vx
            self.y += self.vy

            # Отражение от границ сетки
            if self.x < self.radius or self.x > GRID_X - self.radius:
                self.vx *= -1
            if self.y < self.radius or self.y > GRID_Y - self.radius:
                self.vy *= -1

        elif self.motion_type == 'linear':
            self.x += self.vx
            self.y += self.vy
            if self.x < self.radius or self.x > GRID_X - self.radius:
                self.vx *= -1
            if self.y < self.radius or self.y > GRID_Y - self.radius:
                self.vy *= -1

        elif self.motion_type == 'circular':
            self.angle += self.omega
            self.x = self.cx + self.orbit_radius * np.cos(self.angle)
            self.y = self.cy + self.orbit_radius * np.sin(self.angle)

        else:
            self.vx += rng.uniform(-0.05, 0.05)
            self.vy += rng.uniform(-0.05, 0.05)

            self.vx = np.clip(self.vx, -0.5, 0.5)
            self.vy = np.clip(self.vy, -0.5, 0.5)

            self.x += self.vx
            self.y += self.vy
            if self.x < self.radius or self.x > GRID_X - self.radius:
                self.vx *= -1
            if self.y < self.radius or self.y > GRID_Y - self.radius:
                self.vy *= -1


# Генерация источников
sources = []

sources.append(
    HeatSource(
        x=200,
        y=200,
        power=SOURCE_POWER * 1.5,
        radius=10,
        motion_type='ellipse',
        a=60.0,
        b=120.0,
        omega=0.04,
    )
)

rng = np.random.default_rng()


def thomas(a, b, c, d):
    """Решение трёхдиагональной системы методом прогонки."""
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom

    d_[n - 1] = (d[n - 1] - a[n - 2] * d_[n - 2]) / (
        b[n - 1] - a[n - 2] * c_[n - 2]
    )

    x = np.zeros(n)
    x[-1] = d_[n - 1]

    for i in reversed(range(n - 1)):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


def heat_step(T):
    """Метод переменных направлений для шага теплопроводности."""
    Ny, Nx = T.shape
    r = ALPHA * DT / (2 * DX ** 2)

    ab_x = np.zeros((3, Nx - 2))
    ab_x[0, 1:] = -r
    ab_x[1, :] = 1 + 2 * r
    ab_x[2, :-1] = -r

    ab_y = np.zeros((3, Ny - 2))
    ab_y[0, 1:] = -r
    ab_y[1, :] = 1 + 2 * r
    ab_y[2, :-1] = -r

    # Полушаг 1 (неявно по x)
    rhs = (
        r * T[2:, 1:-1]
        + (1 - 2 * r) * T[1:-1, 1:-1]
        + r * T[:-2, 1:-1]
    )

    T_star_inner = solve_banded((1, 1), ab_x, rhs.T).T

    T_star = T.copy()
    T_star[1:-1, 1:-1] = T_star_inner

    # Полушаг 2 (неявно по y)
    rhs = (
        r * T_star[1:-1, 2:]
        + (1 - 2 * r) * T_star[1:-1, 1:-1]
        + r * T_star[1:-1, :-2]
    )

    T_new_inner = solve_banded((1, 1), ab_y, rhs)

    T_new = T_star.copy()
    T_new[1:-1, 1:-1] = T_new_inner

    return T_new


def add_sources(T, frame):
    """Добавление источников тепла на сетку."""
    for s in sources:
        r = int(s.radius * 3)

        x_min = max(0, int(s.x) - r)
        x_max = min(GRID_X, int(s.x) + r)
        y_min = max(0, int(s.y) - r)
        y_max = min(GRID_Y, int(s.y) + r)

        # Локальная сетка координат вокруг источника
        x = np.arange(x_min, x_max)
        y = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Квадрат расстояния от центра источника
        dist_sq = (xx - s.x) ** 2 + (yy - s.y) ** 2

        # Гауссово распределение
        kernel = np.exp(-dist_sq / (2 * s.radius ** 2))

        T[x_min:x_max, y_min:y_max] += s.power * kernel * DT



DPI = 100
fig = plt.figure(figsize=(GRID_X / DPI, GRID_Y / DPI), dpi=DPI)

ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

im = ax.imshow(
    T, cmap='inferno', vmin=0, vmax=250, interpolation="nearest"
)

plt.tight_layout(pad=0)

ground_truth_data = []  

# Параметры шума
ADD_NOISE = True
NOISE_STD = 30.0 


def update(frame):
    """Функция обновления для анимации."""
    global T
    for i, s in enumerate(sources):
        s.move()
        ground_truth_data.append([frame, i, s.x, s.y])

    add_sources(T, frame)
    T = heat_step(T)
    T *= 0.988

    # Добавление шума для видеопотока
    display_T = np.empty_like(T)
    display_T[:] = T
    if ADD_NOISE:
        noise = np.random.normal(0, NOISE_STD, T.shape)
        display_T += noise
        display_T = np.clip(display_T, 0, 255)

    im.set_data(display_T)
    return [im]


ani = FuncAnimation(fig, update, frames=FRAMES, interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS)
ani.save("heat_simulation.mp4", writer=writer)

# Сохранение координат в CSV
with open('ground_truth.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['frame', 'source_id', 'x', 'y'])
    csv_writer.writerows(ground_truth_data)

print("Данные ground_truth.csv сохранены.")
plt.close(fig)
