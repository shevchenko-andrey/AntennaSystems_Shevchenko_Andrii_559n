import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Вихідні дані (варіант 11)
# -----------------------------
lam = 0.035   # довжина хвилі, м (3.5 см)
ap  = 0.14    # розмір розкриву в H-площині, м
bp  = 0.14    # розмір розкриву в E-площині, м

theta_deg = np.linspace(0, 90, 5000)
theta = np.radians(theta_deg)

LEVEL = 0.707  # рівень половинної потужності

# -----------------------------
# Допоміжні функції
# -----------------------------
def sinc(x):
    y = np.ones_like(x)
    mask = np.abs(x) > 1e-9
    y[mask] = np.sin(x[mask]) / x[mask]
    return y

def half_power_width(theta_deg, F_norm):
    idx = np.where(F_norm <= LEVEL)[0]
    if len(idx) == 0:
        return None, None
    edge = theta_deg[idx[0]]
    return 2*edge, edge

def find_sidelobes(theta_deg, F_norm, n_main_zero=1):
    zero_idx = np.where(F_norm < 1e-3)[0]
    if len(zero_idx) == 0:
        start_idx = 0
    else:
        n_main_zero = min(n_main_zero, len(zero_idx)) - 1
        start_idx = zero_idx[n_main_zero]
    peak_theta = []
    peak_val = []
    for i in range(start_idx + 1, len(F_norm) - 1):
        if F_norm[i] > F_norm[i-1] and F_norm[i] > F_norm[i+1]:
            peak_theta.append(theta_deg[i])
            peak_val.append(F_norm[i])
    return np.array(peak_theta), np.array(peak_val)

# -----------------------------------------------------------
# НУЛІ ВІДПОВІДНО ДО МЕТОДИЧКИ
# -----------------------------------------------------------

# -------- ПЛОЩИНА E: sin(xE)=0 → xE = nπ → θ = arcsin(nλ/bp)
zeros_E = []
for n in range(1,10):
    val = n * lam / bp
    if val < 1:
        zeros_E.append(np.degrees(np.arcsin(val)))

# -------- ПЛОЩИНА H: cos(xH)=0 → xH = π/2(2n+1)
zeros_H = []
for n in range(1,10):
    val = (2*n+1) * lam / (2 * ap)
    if val < 1:
        zeros_H.append(np.degrees(np.arcsin(val)))

zeros_E = np.array(zeros_E)
zeros_H = np.array(zeros_H)

# ============================================================
#                     ПЛОЩИНА E
# ============================================================
xE = np.pi * bp / lam * np.sin(theta)
F_CE  = sinc(xE)
F_E1  = (1 + np.cos(theta)) / 2
F_E   = F_CE * F_E1
F_E_norm = np.abs(F_E) / np.max(np.abs(F_E))

HPBW_E, edge_E = half_power_width(theta_deg, F_E_norm)
theta_sl_E, val_sl_E = find_sidelobes(theta_deg, F_E_norm)

plt.figure(figsize=(10, 6))
plt.plot(theta_deg, F_CE, label="F_CE(θ) — основний множник", linewidth=1.3)
plt.plot(theta_deg, F_E1, label="F_E1(θ) = (1+cosθ)/2", linewidth=1.3)
plt.plot(theta_deg, F_E_norm, label="F_E(θ) — повна ДС", linewidth=1.5)

plt.axhline(LEVEL, color='gray', linestyle='--', linewidth=0.8)
plt.scatter(edge_E, LEVEL, color='red',
            label=f"ШГП E ≈ {HPBW_E:.2f}° (права межа {edge_E:.2f}°)")

# ---- НАНЕСЕННЯ НУЛІВ ----
for z in zeros_E:
    plt.scatter(z, 0, color='black')
plt.plot(zeros_E, np.zeros_like(zeros_E), 'ko', label="Нулі E")

# ---- БІЧНІ ПЕЛЮСТКИ ----
if len(theta_sl_E) > 0:
    for ang, val in zip(theta_sl_E, val_sl_E):
        plt.scatter(ang, val, color='magenta')
    sll1_db = 20*np.log10(val_sl_E[0])
    plt.scatter(theta_sl_E[0], val_sl_E[0], color='cyan',
                label=f"1-й бічний пелюсток E ≈ {sll1_db:.1f} дБ")

plt.title("ДС пірамідального рупору в площині E (варіант 11)")
plt.xlabel("Кут θ, градуси")
plt.ylabel("|F_E(θ)| (нормована)")
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
#                     ПЛОЩИНА H
# ============================================================
xH = np.pi * ap / lam * np.sin(theta)
denH = 1 - (2 * ap / lam * np.sin(theta))**2
denH[np.isclose(denH, 0)] = 1e-9

F_CH  = np.cos(xH) / denH
F_H1  = (1 + np.cos(theta)) / 2
F_H   = F_CH * F_H1
F_H_norm = np.abs(F_H) / np.max(np.abs(F_H))

HPBW_H, edge_H = half_power_width(theta_deg, F_H_norm)
theta_sl_H, val_sl_H = find_sidelobes(theta_deg, F_H_norm)

plt.figure(figsize=(10, 6))
plt.plot(theta_deg, F_CH, label="F_CH(θ) — основний множник", linewidth=1.3)
plt.plot(theta_deg, F_H1, label="F_H1(θ) = (1+cosθ)/2", linewidth=1.3)
plt.plot(theta_deg, F_H_norm, label="F_H(θ) — повна ДС", linewidth=1.5)

plt.axhline(LEVEL, color='gray', linestyle='--', linewidth=0.8)
plt.scatter(edge_H, LEVEL, color='red',
            label=f"ШГП H ≈ {HPBW_H:.2f}° (права межа {edge_H:.2f}°)")

# ---- НУЛІ H ----
for z in zeros_H:
    plt.scatter(z, 0, color='black')
plt.plot(zeros_H, np.zeros_like(zeros_H), 'ko', label="Нулі H")

# ---- БІЧНІ ПЕЛЮСТКИ ----
if len(theta_sl_H) > 0:
    for ang, val in zip(theta_sl_H, val_sl_H):
        plt.scatter(ang, val, color='magenta')
    sll1_db_H = 20*np.log10(val_sl_H[0])
    plt.scatter(theta_sl_H[0], val_sl_H[0], color='cyan',
                label=f"1-й бічний пелюсток H ≈ {sll1_db_H:.1f} дБ")

plt.title("ДС пірамідального рупору в площині H (варіант 11)")
plt.xlabel("Кут θ, градуси")
plt.ylabel("|F_H(θ)| (нормована)")
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
#                ТАБЛИЦІ ДЛЯ ЛАБОРАТОРНОЇ
# ============================================================

print("\nТабл. 1 – Значення нульових кутів (θmin)")
print("-------------------------------------------------")
print("| № | θmin_H | θmin_E |")
for i in range(max(len(zeros_H), len(zeros_E))):
    znH = f"{zeros_H[i]:.2f}" if i < len(zeros_H) else "  -  "
    znE = f"{zeros_E[i]:.2f}" if i < len(zeros_E) else "  -  "
    print(f"| {i+1} | {znH:7} | {znE:7} |")
print("-------------------------------------------------")


print("Табл. 2 – Значення максимальних кутів E (бічні пелюстки)")
print("-------------------------------------------------")
print("| № | θmax_E | FE(θ)  |")
for i in range(len(theta_sl_E)):
    print(f"| {i+1} | {theta_sl_E[i]:7.2f} | {val_sl_E[i]:7.3f} |")
print("-------------------------------------------------")


print("Табл. 3 – Значення максимальних кутів H (бічні пелюстки)")
print("-------------------------------------------------")
print("| № | θmax_H | FH(θ)  |")
for i in range(len(theta_sl_H)):
    print(f"| {i+1} | {theta_sl_H[i]:7.2f} | {val_sl_H[i]:7.3f} |")
print("-------------------------------------------------")

plt.show()