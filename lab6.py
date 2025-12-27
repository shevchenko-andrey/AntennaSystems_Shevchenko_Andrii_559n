import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jn

# -----------------------------
# Вхідні дані (варіант 11)
# -----------------------------
lam = 0.03
D = 0.6
f = 0.2
R0 = D / 2
p = 2 * f
k = 2 * np.pi / lam

v = 3.5 * R0 / p
NORM = 1 / (0.74 * (j1(v)/v) + 0.13)

def J0(x): return j0(x)
def J1(x): return j1(x)
def J2(x): return jn(2, x)

def safe_div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    out = np.zeros_like(a)
    mask = (b != 0)
    out[mask] = a[mask] / b[mask]
    return out

# -----------------------------
# Формули 7.16 і 7.17
# -----------------------------
def F_E(theta):
    u = k * R0 * np.sin(theta)
    denom1 = v**2 - u**2
    denom2 = (1.5 * v)**2 - u**2
    t1 = 0.74 * (v * J1(v) * J0(u) - u * J1(u) * J0(v))
    t1 = safe_div(t1, denom1)
    t2 = 0.26 * safe_div(J1(u), u)
    t3 = 0.25 * (u * J1(u) * J2(1.5 * v) - 1.5 * v * J1(1.5 * v) * J2(u))
    t3 = safe_div(t3, denom2)
    return (np.cos(theta)**2 / 2) * (t1 + t2 + t3) * NORM

def F_H(theta):
    u = k * R0 * np.sin(theta)
    denom1 = v**2 - u**2
    denom2 = (1.5 * v)**2 - u**2
    t1 = 0.74 * (v * J1(v) * J0(u) - u * J1(u) * J0(v))
    t1 = safe_div(t1, denom1)
    t2 = 0.26 * safe_div(J1(u), u)
    t3 = 0.25 * (u * J1(u) * J2(1.5 * v) - 1.5 * v * J1(1.5 * v) * J2(u))
    t3 = safe_div(t3, denom2)
    return (np.cos(theta)**2 / 2) * (t1 + t2 - t3) * NORM

# -----------------------------
# Основні обчислення
# -----------------------------
theta_deg = np.linspace(0, 90, 4000)
theta = np.radians(theta_deg)

FE_raw = F_E(theta)
FH_raw = F_H(theta)

# Нормування (амплітуда в 0° = 1)
FE = FE_raw / np.abs(FE_raw[0])
FH = FH_raw / np.abs(FH_raw[0])

x_axis = k * R0 * np.sin(theta)

# -----------------------------
# ПОШУК НУЛІВ
# -----------------------------
def find_zeros(F):
    zeros = []
    for i in range(len(F)-1):
        if F[i] * F[i+1] < 0:
            zeros.append(i if abs(F[i]) < abs(F[i+1]) else i+1)
    return np.array(zeros)

zerosE = find_zeros(FE_raw)
zerosH = find_zeros(FH_raw)

# -----------------------------
# ПОШУК МАКСИМУМІВ
# -----------------------------
def find_maxima(F):
    dF = np.diff(F)
    return np.where((dF[:-1] > 0) & (dF[1:] < 0))[0] + 1

maxE = find_maxima(FE_raw)
maxH = find_maxima(FH_raw)

# -----------------------------
# HPBW
# -----------------------------
half = 0.707

idx_E = np.where(FE >= half)[0]
idx_H = np.where(FH >= half)[0]

#точка HPBW
HPBW_E = idx_E[-1]
HPBW_H = idx_H[-1]

HP_E_x = x_axis[HPBW_E]
HP_E_y = half

HP_H_x = x_axis[HPBW_H]
HP_H_y = half

# -----------------------------
# ГРАФІК
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(x_axis, FE, label="E-площина", color='orange')
plt.plot(x_axis, FH, label="H-площина", color='teal')

# НУЛІ
plt.scatter(x_axis[zerosE], FE[zerosE], color='black', marker='x', s=50)
plt.scatter(x_axis[zerosH], FH[zerosH], color='black', marker='x', s=50)

# МАКСИМУМИ
plt.scatter(x_axis[maxE], FE[maxE], color='red', s=30)
plt.scatter(x_axis[maxH], FH[maxH], color='blue', s=30)

# HPBW
plt.scatter(HP_E_x, HP_E_y, color='orange', edgecolors='black', s=90)
plt.scatter(HP_H_x, HP_H_y, color='teal', edgecolors='black', s=90)

plt.axhline(0.707, linestyle="--", color="gray", label="Рівень 0.707")

plt.title("ДС дзеркальної антени у координаті kR₀·sin(θ)\n(формули 7.16 та 7.17)")
plt.xlabel("kR₀·sin(θ)")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.legend()
plt.ylim(-0.2, 1.1)

# -----------------------------
# ТАБЛИЦІ
# -----------------------------
print("\nТаблиця 1 — Нульові значення (E та H)")
print("---------------------------------------------")

m = 1
for idx in zerosE:
    print(f"{m:2d}. E  θ={theta_deg[idx]:8.3f}°   u={x_axis[idx]:8.4f}")
    m += 1
for idx in zerosH:
    print(f"{m:2d}. H  θ={theta_deg[idx]:8.3f}°   u={x_axis[idx]:8.4f}")
    m += 1

print("\nТаблиця 2 — Максимальні значення (E)")
m = 1
for idx in maxE:
    print(f"{m:2d}. θ={theta_deg[idx]:8.3f}°   u={x_axis[idx]:8.4f}   F={FE[idx]:6.4f}")
    m += 1

print("\nТаблиця 3 — Максимальні значення (H)")
m = 1
for idx in maxH:
    print(f"{m:2d}. θ={theta_deg[idx]:8.3f}°   u={x_axis[idx]:8.4f}   F={FH[idx]:6.4f}")
    m += 1

# ============================================================
#  РОЗРАХУНОК HPBW (ширини головної пелюстки)
# ============================================================

def compute_hpbw(F, theta_deg):
    half = 0.707

    # знаходимо праву точку перетину з рівнем 0.707
    idx = np.where(F >= half)[0]
    theta_half = theta_deg[idx[-1]]  # права точка

    # HPBW = повна ширина головної пелюстки
    return 2 * theta_half, theta_half


HPBW_E_full, HPBW_E_right = compute_hpbw(FE, theta_deg)
HPBW_H_full, HPBW_H_right = compute_hpbw(FH, theta_deg)

print("\n================== HPBW ==================")
print(f"HPBW (E-площина): {HPBW_E_full:.3f}°  (права точка: {HPBW_E_right:.3f}°)")
print(f"HPBW (H-площина): {HPBW_H_full:.3f}°  (права точка: {HPBW_H_right:.3f}°)")


# ============================================================
#  РОЗРАХУНОК SLL (side lobe level — рівень бокового пелюстка)
# ============================================================

def compute_sll(F, theta_deg, zeros):
    first_zero = zeros[0]

    # беремо ділянку після першого нуля
    F_seg = F[first_zero+1:]
    dF = np.diff(F_seg)

    # шукаємо локальні максимуми
    maxima = np.where((dF[:-1] > 0) & (dF[1:] < 0))[0] + 1

    if len(maxima) == 0:
        return None, None

    # найбільший пелюсток
    idx_loc = maxima[np.argmax(F_seg[maxima])]
    idx_global = first_zero + 1 + idx_loc

    amplitude = F[idx_global]
    angle = theta_deg[idx_global]

    SLL_dB = 20 * np.log10(abs(amplitude))

    return SLL_dB, angle


SLL_E_dB, SLL_E_angle = compute_sll(FE, theta_deg, zerosE)
SLL_H_dB, SLL_H_angle = compute_sll(FH, theta_deg, zerosH)

print("\n================== SLL ==================")
print(f"SLL (E): {SLL_E_dB:.2f} дБ при куті {SLL_E_angle:.3f}°")
print(f"SLL (H): {SLL_H_dB:.2f} дБ при куті {SLL_H_angle:.3f}°")

plt.show()