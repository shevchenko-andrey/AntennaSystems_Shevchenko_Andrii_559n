import numpy as np
import matplotlib.pyplot as plt

# ------------ 1. Вхідні данні-------------
eps_r = 2.2
tg_delta = 2e-4

lambda_cm = 4.3
h_cm      = 10.0
l_cm      = 23.7

d_max_cm = 2.9
d_min_cm = 1.7
d_cp_cm  = 2.3

# Переклад в метры
lam = lambda_cm / 100
h   = h_cm / 100
l   = l_cm / 100

# Коэфіцієнт уповільнення (4.7)
xi = 1 + lam / (2 * l)
print("xi =", xi)

# ------------ 2. Кутова сітка ----------------
theta_deg = np.linspace(0, 90, 2001)
theta = np.deg2rad(theta_deg)

# ------------ 3. Формули ДС -------------------

# (4.3) бігуча хвиля F_B
def F_B(theta):
    k = np.pi * (l / lam)
    X = xi - np.cos(theta)

    num = np.sin(k * X)
    den = X * np.sin(k * (xi - 1))

    FB = np.zeros_like(theta)
    mask = np.abs(den) > 1e-12
    FB[mask] = num[mask] / den[mask]
    FB[~mask] = 1.0
    return FB

# (4.18) множник грат F_C
def F_C(theta):
    return np.cos(np.pi * h / lam * np.sin(theta))

FB = F_B(theta)
FC = F_C(theta)

# Один стержень
F_H1 = FB
F_E1 = FB * np.cos(theta)

# Два стержня
F_H2 = FB * FC
F_E2 = F_H2 * np.cos(theta)

# ------------ 4. Нормування -------------------
def norm(x):
    return np.abs(x) / np.max(np.abs(x))

# ------------ 5. Пошук нулів та максимумів -----
def find_zeros(f):
    idx = []
    for i in range(1, len(f)):
        if f[i-1] * f[i] < 0:
            idx.append(i)
    return idx

def find_maxima(f):
    f = np.abs(f)
    idx = []
    for i in range(1, len(f)-1):
        if f[i] > f[i-1] and f[i] > f[i+1]:
            idx.append(i)
    return idx

level = 1 / np.sqrt(2)

def find_hpbw_indices(f, zeros_idx, level=level):
    """
    Находит две точки пересечения нормированной ДС с уровнем 0.707
    в пределах главного лепестка (между максимумом и первым нулём).
    Возвращает (i_left, i_right, θ_left, θ_right, width_deg)
    """
    fn = norm(f)

    # главный лепесток — до первого нуля
    if zeros_idx:
        end = zeros_idx[0]
    else:
        end = len(fn)

    left_idx = None
    right_idx = None

    # поиск точек пересечения линии 0.707
    for i in range(1, end):
        if (fn[i-1] - level) * (fn[i] - level) <= 0:
            if left_idx is None:
                left_idx = i
            else:
                right_idx = i
                break

    if left_idx is None or right_idx is None:
        return None

    θ_left = theta_deg[left_idx]
    θ_right = theta_deg[right_idx]
    width = θ_right - θ_left

    return left_idx, right_idx, θ_left, θ_right, width


# Одностержневая
zeros_H1 = find_zeros(F_H1)
zeros_E1 = zeros_H1[:]

max_H1 = find_maxima(F_H1)
max_E1 = find_maxima(F_E1)

# Двухстержневая
zeros_H2 = find_zeros(F_H2)
zeros_E2 = find_zeros(F_E2)

max_H2 = find_maxima(F_H2)
max_E2 = find_maxima(F_E2)



# ============================================================
#                ГРАФИК 1 — Одностержневая ДС (H та E)
# ============================================================

plt.figure(figsize=(10, 6))

FBn = norm(F_H1)
FEn = norm(F_E1)
cosn = norm(np.cos(theta))
level = 1 / np.sqrt(2)

plt.plot(theta_deg, FBn, 'k--', label='H-площина |F_B|')
plt.plot(theta_deg, cosn, 'g-',  label='cosθ')
plt.plot(theta_deg, FEn, 'b-',   label='E-площина |F_E|')

plt.axhline(level, linestyle='--', color='gray', label="Рівень 0.707")

ax = plt.gca()

# ----------------------------------------------------------
#  ПОИСК ПЕРЕСЕЧЕНИЯ УРОВНЯ 0.707 (одна точка!)
# ----------------------------------------------------------
def find_hpbw_point(f, theta_deg, level=1/np.sqrt(2)):
    fn = norm(f)
    for i in range(1, len(fn)):
        if fn[i-1] >= level and fn[i] <= level:
            # линейная интерполяция
            x1, x2 = theta_deg[i-1], theta_deg[i]
            y1, y2 = fn[i-1], fn[i]
            x_cross = x1 + (level - y1) * (x2 - x1) / (y2 - y1)
            return x_cross
    return None

theta_H_lvl = find_hpbw_point(F_H1, theta_deg)
theta_E_lvl = find_hpbw_point(F_E1, theta_deg)

# ----- точка для H-плоскости -----
if theta_H_lvl:
    plt.scatter(theta_H_lvl, level, color='black', s=60)
    plt.text(theta_H_lvl + 1, level + 0.025, f"{theta_H_lvl:.2f}°",
             color='black', fontsize=9)

# ----- точка для E-плоскости -----
if theta_E_lvl:
    plt.scatter(theta_E_lvl, level, color='blue', s=60)
    plt.text(theta_E_lvl + 1, level - 0.05, f"{theta_E_lvl:.2f}°",
             color='red', fontsize=9)


# ----------------------------------------------------------
#  НУЛИ H
# ----------------------------------------------------------
for i, idx in enumerate(zeros_H1[:5], 1):
    x = theta_deg[idx]
    plt.scatter(x, 0, color='orange', s=40)
    ax.text(x, -0.10, f'θ0H{i}', color='black', ha='center',
            transform=ax.get_xaxis_transform())

# ----------------------------------------------------------
#  НУЛИ E
# ----------------------------------------------------------
for i, idx in enumerate(zeros_E1[:5], 1):
    x = theta_deg[idx]
    plt.scatter(x, 0, color='purple', marker='s', s=40)
    ax.text(x, -0.13, f'θ0E{i}', color='blue', ha='center',
            transform=ax.get_xaxis_transform())

# ----------------------------------------------------------
#  МАКСИМУМЫ H
# ----------------------------------------------------------
for i, idx in enumerate(max_H1[:5], 1):
    x = theta_deg[idx]
    y = FBn[idx]
    plt.scatter(x, y, color='black', marker='x')
    plt.text(x+1, y+0.03, f'θmH{i}', color='black')

# ----------------------------------------------------------
#  МАКСИМУМЫ E
# ----------------------------------------------------------
for i, idx in enumerate(max_E1[:5], 1):
    x = theta_deg[idx]
    y = FEn[idx]
    plt.scatter(x, y, color='blue', marker='*')
    plt.text(x+1, y+0.03, f'θmE{i}', color='blue')


plt.title("Однострижнева ДС (H та E)")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.ylim(0, 1.1)
plt.xlim(0, 90)
plt.legend()


plt.title("Однострижнева ДС (H та E)")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.ylim(0, 1.1)
plt.xlim(0, 90)
plt.legend()

# ============================================================
#                     ГРАФИК 2 — Двострижнева ДС (H-площина)
# ============================================================

plt.figure(figsize=(10, 6))

H2n = norm(F_H2)
FBn = norm(FB)
FCn = norm(FC)
level = 1 / np.sqrt(2)

plt.plot(theta_deg, H2n, 'b-', label='|F_H(θ)| для 2 стрижнів')
plt.plot(theta_deg, FBn, 'k--', label='|F_B(θ)| (один стрижень)')
plt.plot(theta_deg, FCn, 'g-', label='|F_C(θ)| (множник грат)')
plt.axhline(level, linestyle='--', color='gray', label='Рівень 0.707')

# ----------------------------------------------------------
#  ТОЧКА пересечения уровня 0.707
# ----------------------------------------------------------
theta_H2_lvl = find_hpbw_point(F_H2, theta_deg)
if theta_H2_lvl:
    plt.scatter(theta_H2_lvl, level, color='blue', s=70, zorder=5)
    plt.text(theta_H2_lvl + 1, level + 0.03,
             f'{theta_H2_lvl:.2f}°', color='blue')

ax = plt.gca()

# ----------------------------------------------------------
#  НУЛИ F_H2 (основной ДС)
# ----------------------------------------------------------
for i, idx in enumerate(zeros_H2[:10], 1):
    x = theta_deg[idx]
    plt.scatter(x, 0, color='orange', s=45)
    ax.text(x, -0.10, f'θ0H{i}', color='blue', ha='center',
            transform=ax.get_xaxis_transform())


# ----------------------------------------------------------
#  МАКСИМУМЫ F_H2
# ----------------------------------------------------------
for i, idx in enumerate(max_H2[:10], 1):
    x = theta_deg[idx]
    y = H2n[idx]
    plt.scatter(x, y, color='black', marker='x')
    plt.text(x + 1, y + 0.025, f'θmH{i}', color='blue')


# ==========================================================
#  ДОБАВЛЯЕМ НУЛИ И МАКСИМУМЫ ДЛЯ F_B (один стрижень)
# ==========================================================

plt.title("Двострижнева ДС — H-площина")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.ylim(0, 1.15)
plt.xlim(0, 90)
plt.legend()

# ============================================================
#                     ГРАФИК 3 — Двострижнева ДС (E-площина)
# ============================================================

plt.figure(figsize=(10, 6))

E2n = norm(F_E2)
H2n = norm(F_H2)
FBn = norm(FB)
FCn = norm(FC)
cosn = norm(np.cos(theta))
level = 1 / np.sqrt(2)

plt.plot(theta_deg, FBn, 'k--',  label='|F_B(θ)| (один стрижень)')
plt.plot(theta_deg, FCn, 'g-',   label='|F_C(θ)| (множник грат)')
plt.plot(theta_deg, H2n, 'b-',   label='|F_H(θ)| = |F_B·F_C|')
plt.plot(theta_deg, E2n, 'r-', linewidth=2, label='|F_E(θ)| = |F_B·F_C·cosθ|')
plt.plot(theta_deg, cosn, 'm-', label='|cos θ|')
plt.axhline(level, linestyle='--', color='gray', label='Рівень 0.707')

ax = plt.gca()

# ----------------------------------------------------------
#  ТОЧКА пересечения уровня 0.707 (для F_E2)
# ----------------------------------------------------------
theta_E2_lvl = find_hpbw_point(F_E2, theta_deg)
if theta_E2_lvl:
    plt.scatter(theta_E2_lvl, level, color='red', s=70)
    plt.text(theta_E2_lvl + 1, level + 0.03,
             f'{theta_E2_lvl:.2f}°', color='red')


# ----------------------------------------------------------
#  НУЛИ F_E2 (главная ДС)
# ----------------------------------------------------------
for i, idx in enumerate(zeros_E2[:10], 1):
    x = theta_deg[idx]
    plt.scatter(x, 0, color='purple', s=50)
    ax.text(x, -0.07, f'θ0E{i}', color='red', ha='center',
            transform=ax.get_xaxis_transform())


# ----------------------------------------------------------
#  МАКСИМУМЫ F_E2
# ----------------------------------------------------------
for i, idx in enumerate(max_E2[:10], 1):
    x = theta_deg[idx]
    y = E2n[idx]
    plt.scatter(x, y, color='red', marker='*', s=80)
    plt.text(x + 1, y + 0.01, f'θmE{i}', color='red')


# ==========================================================
#  НУЛИ И МАКСИМУМЫ F_H2  (двухстрижневая H)
# ==========================================================

# ---- Нули ----
for i, idx in enumerate(zeros_H2[:10], 1):
    x = theta_deg[idx]
    plt.scatter(x, 0, color='orange', s=45)
    ax.text(x, -0.10, f'θ0H{i}', color='blue', ha='center',
            transform=ax.get_xaxis_transform())

# ---- Максимумы ----
for i, idx in enumerate(max_H2[:10], 1):
    x = theta_deg[idx]
    y = H2n[idx]
    plt.scatter(x, y, color='blue', marker='x')
    plt.text(x + 1, y + 0.03, f'θmH{i}', color='blue')

plt.title("Двострижнева ДС — E-площина")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.ylim(0, 1.15)
plt.xlim(0, 90)
plt.legend()
# ============================================================
#              ТАБЛИЦІ ДЛЯ ВСІХ ТОЧОК НА ГРАФІКАХ
# ============================================================

def print_table(title, angles, values=None):
    print("\n" + title)
    print("--------------------------------------")
    print("| № |   θ (град)   |   Значення      |")
    print("--------------------------------------")
    for i, ang in enumerate(angles, 1):
        if values is None:
            print(f"| {i:2d} |   {ang:.4f}    |     0.0000      |")
        else:
            print(f"| {i:2d} |   {ang:.4f}    |    {values[i-1]:.4f}    |")
    print("--------------------------------------")


# ======== 1. ОДНОСТРИЖНЕВА АНТЕНА ===========================
print_table("ТАБЛ. 1 — Нульові кути H₁",
            [theta_deg[i] for i in zeros_H1[:10]])

print_table("ТАБЛ. 2 — Нульові кути E₁",
            [theta_deg[i] for i in zeros_E1[:10]])

print_table("ТАБЛ. 3 — Максимуми H₁",
            [theta_deg[i] for i in max_H1[:10]],
            [norm(F_H1)[i] for i in max_H1[:10]])

print_table("ТАБЛ. 4 — Максимуми E₁",
            [theta_deg[i] for i in max_E1[:10]],
            [norm(F_E1)[i] for i in max_E1[:10]])


# ======== 2. ДВОСТРИЖНЕВА АНТЕНА ============================
print_table("ТАБЛ. 5 — Нульові кути H₂",
            [theta_deg[i] for i in zeros_H2[:10]])

print_table("ТАБЛ. 6 — Нульові кути E₂",
            [theta_deg[i] for i in zeros_E2[:10]])

print_table("ТАБЛ. 7 — Максимуми H₂",
            [theta_deg[i] for i in max_H2[:10]],
            [norm(F_H2)[i] for i in max_H2[:10]])

print_table("ТАБЛ. 8 — Максимуми E₂",
            [theta_deg[i] for i in max_E2[:10]],
            [norm(F_E2)[i] for i in max_E2[:10]])


# ======== 3. ТОЧКИ ПЕРЕТИНУ РІВНЯ 0.707 ====================
print("\nТАБЛ. 9 — Кути на рівні 0.707")
print("--------------------------------------")
print("| Графік           | θ (град)        |")
print("--------------------------------------")

if theta_H_lvl:
    print(f"| H₁                |  {theta_H_lvl:.4f} |")

if theta_E_lvl:
    print(f"| E₁                |  {theta_E_lvl:.4f} |")

if theta_H2_lvl:
    print(f"| H₂                |  {theta_H2_lvl:.4f} |")

if theta_E2_lvl:
    print(f"| E₂                |  {theta_E2_lvl:.4f} |")

print("--------------------------------------")

plt.show()