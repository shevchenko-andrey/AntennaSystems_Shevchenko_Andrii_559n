import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# ВХІДНІ ДАНІ (Варіант 11)
# -----------------------------
lam = 0.03          # довжина хвилі λ = 3 см
d = 0.04            # відстань між щілинами = 4 см
N = 5               # кількість щілин (тип 2 по табл. 4)

k = 2 * np.pi / lam

# Кутова сітка: θ ∈ [–90°, +90°]
theta_deg = np.linspace(-90, 90, 2000)
theta = np.deg2rad(theta_deg)

# ----------------------------------
# ЕЛЕМЕНТНА ДІАГРАМА
# ----------------------------------
def F_element(theta):
    return np.abs(np.cos(theta))

# ----------------------------------
# МНОЖНИК РЕШІТКИ
# ----------------------------------
def array_factor(theta):
    psi = k * d * np.sin(theta)
    psi_half = psi / 2

    num = np.sin(N * psi_half)
    den = N * np.sin(psi_half)
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)

    return np.abs(num / den)

# ----------------------------------
# ПОВНІ ДІАГРАМИ
# ----------------------------------
F_el = F_element(theta)
AF = array_factor(theta)

F_H = F_el * AF
F_H_norm = F_H / np.max(F_H)

F_E = F_el * np.abs(np.cos(theta)) * AF
F_E_norm = F_E / np.max(F_E)

F_el_norm = F_el / np.max(F_el)
AF_norm = AF / np.max(AF)

# ----------------------------------
# HPBW
# ----------------------------------
def HPBW(pattern, theta):
    patt = pattern / np.max(pattern)
    target = 1 / np.sqrt(2)

    i0 = np.argmax(patt)

    left = None
    for i in range(i0, 0, -1):
        if patt[i] >= target and patt[i-1] < target:
            left = theta[i]
            break

    right = None
    for i in range(i0, len(patt)-1):
        if patt[i] >= target and patt[i+1] < target:
            right = theta[i]
            break

    if left is None or right is None:
        return None

    return np.degrees(right - left)

# ----------------------------------
# ПЕРШИЙ БОКОВИЙ ПЕЛЮСТОК
# ----------------------------------
def first_sidelobe_level(pattern, theta):
    patt = pattern / np.max(pattern)
    i0 = np.argmax(patt)
    start = i0 + 20
    sub = patt[start:]
    idx = start + np.argmax(sub)

    angle = np.degrees(theta[idx])
    level_db = 20 * np.log10(patt[idx])
    return angle, level_db

# ----------------------------------
# ЛОКАЛЬНІ МАКСИМУМИ (для графіків)
# ----------------------------------
def local_maxima_indices(y):
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1

def annotate_peaks(theta_deg, y_norm, min_height=0.05, exclude_main_deg=3.0):
    idxs = local_maxima_indices(y_norm)
    for i in idxs:
        if y_norm[i] >= min_height and abs(theta_deg[i]) >= exclude_main_deg:
            plt.plot(theta_deg[i], y_norm[i], "go", markersize=4)
            plt.text(
                theta_deg[i],
                y_norm[i] + 0.02,
                f"θ={theta_deg[i]:.1f}°",
                fontsize=8,
                ha="center",
                color="green"
            )

# ----------------------------------
# НУЛЬОВІ КУТИ
# ----------------------------------
def find_zeros(theta_deg, patt, eps=1e-3):
    zeros = []
    for i in range(1, len(patt)):
        if patt[i-1] > eps and patt[i] <= eps:
            zeros.append(theta_deg[i])
    return zeros

# ----------------------------------
# МАКСИМУМИ БОКОВИХ ПЕЛЮСТОК (для таблиць)
# ----------------------------------
def find_sidelobe_maxima(theta_deg, patt):
    maxima = []
    for i in range(1, len(patt)-1):
        if patt[i] > patt[i-1] and patt[i] > patt[i+1]:
            if abs(theta_deg[i]) >= 3:
                maxima.append((theta_deg[i], patt[i]))
    return maxima

# ----------------------------------
# РОЗРАХУНКИ + ВИВІД
# ----------------------------------
hpbw_H = HPBW(F_H_norm, theta)
hpbw_E = HPBW(F_E_norm, theta)

theta_sl_H, sl_H_db = first_sidelobe_level(F_H_norm, theta)
theta_sl_E, sl_E_db = first_sidelobe_level(F_E_norm, theta)

print(f"HPBW (H-площина): {hpbw_H:.2f} град")
print(f"HPBW (E-площина): {hpbw_E:.2f} град")
print(f"Перший боковий пелюсток H: кут ≈ {theta_sl_H:.1f}°, рівень ≈ {sl_H_db:.1f} дБ")
print(f"Перший боковий пелюсток E: кут ≈ {theta_sl_E:.1f}°, рівень ≈ {sl_E_db:.1f} дБ")

# ----------------------------------
# ТАБЛИЦЯ 1 — НУЛЬОВІ КУТИ
# ----------------------------------
zeros = find_zeros(theta_deg, F_H_norm)

print("\nТаблиця 1 — Нульові кути")
print("---------------------------------")
print("| № |   θ₀ [°]  |")
for i, th in enumerate(zeros, 1):
    print(f"| {i:2d} | {th:8.2f} |")
print("---------------------------------")

# ----------------------------------
# ТАБЛИЦЯ 2 — БОКОВІ ПЕЛЮСТКИ
# ----------------------------------
sidelobes = find_sidelobe_maxima(theta_deg, F_H_norm)

print("\nТаблиця 2 — Максимальні кути бокових пелюсток")
print("-------------------------------------------")
print("| № |  θₘ [°]  |  F_H(θₘ) |")
for i, (th, val) in enumerate(sidelobes, 1):
    print(f"| {i:2d} | {th:8.2f} | {val:8.4f} |")
print("-------------------------------------------")

# ----------------------------------
# ГРАФІКИ
# ----------------------------------
level_707 = 1 / np.sqrt(2)

plt.figure(figsize=(12, 7))
plt.plot(theta_deg, F_el_norm, label="Елементна ДС F1H(θ)")
plt.plot(theta_deg, AF_norm, "--", label="Множник решітки FHC(θ)")
plt.plot(theta_deg, F_H_norm, label="Повна ДС F_H(θ)")
plt.axhline(level_707, linestyle="--", label="Рівень 0.707")
annotate_peaks(theta_deg, F_H_norm)
plt.xlabel("θ, градуси")
plt.ylabel("Нормована амплітуда")
plt.title("Діаграма спрямованості ХвЩА (H-площина, варіант 11)")
plt.grid(True)
plt.legend()
plt.ylim(0, 1.1)
plt.xlim(-90, 90)

plt.figure(figsize=(12, 7))
plt.plot(theta_deg, F_el_norm, label="Елементна ДС F1E(θ)")
plt.plot(theta_deg, AF_norm, "--", label="Множник решітки FEC(θ)")
plt.plot(theta_deg, F_E_norm, label="Повна ДС F_E(θ)")
plt.axhline(level_707, linestyle="--", label="Рівень 0.707")
annotate_peaks(theta_deg, F_E_norm)
plt.xlabel("θ, градуси")
plt.ylabel("Нормована амплітуда")
plt.title("Діаграма спрямованості ХвЩА (E-площина, варіант 11)")
plt.grid(True)
plt.legend()
plt.ylim(0, 1.1)
plt.xlim(-90, 90)

plt.show()
