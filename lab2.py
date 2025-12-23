import numpy
import matplotlib.pyplot as plt
import math


# ==========================================================
# ВХІДНІ ДАНІ ДЛЯ ВАРІАНТА №11
# F = 620 МГц, N = 6
# ==========================================================
N = 6  # Число елементів антени
F = 620 * 10 ** 6  # Частота (620 МГц)
C = 299792458  # Швидкість світла
# ==========================================================

F1E = [1]
FC = [1]
FE = [1]
steps = [0]
SGP1 = 0
SGP2 = 0
fS1 = 0
fS2 = 0
Zeros = []

max_x_FC = []
max_y_FC = []
max_x_FE = []
max_y_FE = []

min_x_FC = []
min_y_FC = []
min_x_FE = []
min_y_FE = []

lambd = C / F
print(f"λ (довжина хвилі) = {lambd:.4f} (m)")
d = 0.25 * lambd
print(f"dcp (крок елементів) = {d:.4f} (m)")
k = (2 * numpy.pi) / lambd
print(f"k (хвильове число) = {k:.4f} (rad/m)")

# Ітерація кута theta від 0 до 90 градусів
for teta in numpy.arange(0.01, numpy.pi / 2, 0.00001):
    # Множник елемента F1e(theta) - Площина Е (випромінювання диполя)
    mn1 = abs((numpy.cos(numpy.pi / 2 * numpy.sin(teta)) / numpy.cos(teta)))

    # Множник решітки FH(theta) - Площина H
    arg_H = (k * d * (1 - numpy.cos(teta))) / 2
    mn2 = abs(numpy.sin(N * arg_H) / (N * numpy.sin(arg_H)))

    # Загальна ДС FE(theta) - Площина Е
    mn3 = mn1 * mn2

    F1E.append(mn1)
    FC.append(mn2)
    FE.append(mn3)

    teta_deg = math.degrees(teta)

    # Визначення кутів для рівня 0.707 (половинна потужність)
    if 0.707 < mn2 < 0.708 and SGP1 == 0:
        SGP1 = 2 * teta_deg
        fS1 = mn2

    if 0.707 < mn3 < 0.708 and SGP2 == 0:
        SGP2 = 2 * teta_deg
        fS2 = mn3

    steps.append(teta_deg)

# Пошук екстремумів (бічних пелюсток)
for i in range(1, len(FC) - 1):
    # Максимуми
    if FC[i] > FC[i - 1] and FC[i] > FC[i + 1]:
        max_x_FC.append(steps[i])
        max_y_FC.append(FC[i])
    if FE[i] > FE[i - 1] and FE[i] > FE[i + 1]:
        max_x_FE.append(steps[i])
        max_y_FE.append(FE[i])

    # Мінімуми
    if FC[i] < FC[i - 1] and FC[i] < FC[i + 1]:
        min_x_FC.append(steps[i])
        min_y_FC.append(FC[i])
    if FE[i] < FE[i - 1] and FE[i] < FE[i + 1]:
        min_x_FE.append(steps[i])
        min_y_FE.append(FC[i])  # У площині E нулі визначаються переважно нулями FC

print(f"-----------------------------------")
print(f"Ширина головної пелюстки (ШГП) на рівні 0.707:")
print(f"  В площині H = {round(SGP1, 2)}°")
print(f"  В площині E = {round(SGP2, 2)}°")
print(f"-----------------------------------")

# Виведення загальної таблиці аналізу ДС
len_value = [len(max_x_FC), len(max_x_FE), len(min_x_FC), len(min_x_FE)]
max_len = max(len_value)
print("Табл. 1 - Аналіз ДС Директорної антени в площині Н та Е")
print("---------------------------------------------------------")
print("| № |θ_min_H|θ_min_E|θ_max_H| F_H(θ) |θ_max_E| F_E(θ) |")
print("---------------------------------------------------------")
for i in range(0, max_len):
    v1 = f" {i + 1:.0f}"
    v2 = f"{min_x_FC[i]:.2f}" if i < len(min_x_FC) else '  -  '
    v3 = f"{min_x_FE[i]:.2f}" if i < len(min_x_FE) else '  -  '
    v4 = f"{max_x_FC[i]:.2f}" if i < len(max_x_FC) else '  -  '
    h4 = f"{max_y_FC[i]:.3f}" if i < len(max_x_FC) else '  -  '
    v5 = f"{max_x_FE[i]:.2f}" if i < len(max_x_FE) else '  -  '
    h5 = f"{max_y_FE[i]:.3f}" if i < len(max_x_FE) else '  -  '
    print(f"|{v1:^2}|{v2:^7}|{v3:^7}|{v4:^7}|{h4:^8}|{v5:^7}|{h5:^8}|")
print("---------------------------------------------------------")

# ==========================================================
# ГЕНЕРАЦІЯ ГРАФІКА ДІАГРАМ СПРЯМОВАНОСТІ
# ==========================================================

fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
ax.plot(steps, F1E, linewidth=0.7, label="$ F_{1e}(θ) $ - Множник елемента")
ax.plot(steps, FC, linewidth=0.7, label="$ F_{H}(θ) $ - Площина H (Множник решітки)")
ax.plot(steps, FE, linewidth=0.7, label="$ F_{E}(θ) $ - Площина E (Загальна ДС)")

# Позначки ШГП (рівень 0.707)
ax.plot(SGP1 / 2, fS1, 'ro', markersize=4, label=f"ШГП в H ({round(SGP1, 2)}°)")
ax.plot(SGP2 / 2, fS2, 'go', markersize=4, label=f"ШГП в E ({round(SGP2, 2)}°)")

# Анотації для ШГП
plt.annotate(f'({SGP1 / 2:.2f}\u00b0)',
             xy=(SGP1 / 2, fS1),
             xytext=((SGP1 / 2) + 3, fS1 - 0.1),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

plt.annotate(f'({SGP2 / 2:.2f}\u00b0)',
             xy=(SGP2 / 2, fS2),
             xytext=((SGP2 / 2) - 12, fS2 + 0.05),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=6)

# Пунктирні лінії для рівня 0.707
ax.hlines(y=fS1, xmin=0, xmax=SGP1 / 2, colors='r', linestyles='--', linewidth=0.5)
ax.vlines(x=SGP1 / 2, ymin=0, ymax=fS1, colors='r', linestyles='--', linewidth=0.5)
ax.vlines(x=SGP2 / 2, ymin=0, ymax=fS2, colors='g', linestyles='--', linewidth=0.5)

# Позначки екстремумів
ax.plot(max_x_FC, max_y_FC, "o", markersize=4, color="black", label="$ \\theta_{max} $ FH")
ax.plot(max_x_FE, max_y_FE, "o", markersize=4, color="grey", label="$ \\theta_{max} $ FE")
ax.plot(min_x_FC, min_y_FC, "o", markersize=4, color="blue", label="$ \\theta_{min} $")

# Налаштування вигляду
ax.set_xlabel('Кут відносно осі антени, $θ$ (°)', fontsize=10)
ax.set_ylabel('Нормована ДС, $|F(θ)|$', fontsize=10)
plt.title(f'Нормовані ДС Директорної антени (Варіант 11: N={N}, F=620 МГц)', fontsize=12)
plt.xticks(numpy.arange(0, 100, 5), fontsize=7)
plt.yticks(numpy.arange(0, 1.2, 0.1), fontsize=7)
plt.ylim(-0.01, 1.05)
plt.xlim(0, 90.5)
plt.legend(loc="upper right", fontsize=7)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')

# Збереження та відображення
fig.savefig(f"ДС_Director_Var11_N{N}_F620.jpg", dpi=600)
plt.show()