import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load 30-run Central Composite Design (CCD) dataset
df = pd.read_csv("ccd_data.csv")

A = df["Dosage"].values       # g/L
B = df["Conc"].values         # mg/L
C = df["Time"].values         # min
D = df["pH"].values
y = df["Degradation"].values  # %

# Quadratic response surface design matrix (15 terms)
X = np.column_stack([
    np.ones_like(A),
    A, B, C, D,
    A*B, A*C, A*D, B*C, B*D, C*D,
    A**2, B**2, C**2, D**2
])

terms = ["Intercept", "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "A2", "B2", "C2", "D2"]

# Least-squares regression
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ beta

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
n, p = X.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)

print(f"R2 = {r2:.4f} | Adjusted R2 = {adj_r2:.4f}\n")
for name, coef in zip(terms, beta):
    print(f"{name:<10}: {coef:10.4f}")

# Plot 1: Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(y, y_pred, color="#1f4e78", edgecolor="black", alpha=0.85)
lims = [min(y.min(), y_pred.min()) - 3, max(y.max(), y_pred.max()) + 3]
axes[0].plot(lims, lims, "r--", linewidth=1.2)
axes[0].set_xlabel("Actual Degradation (%)")
axes[0].set_ylabel("Predicted Degradation (%)")
axes[0].set_title(f"Predicted vs. Actual (R² = {r2:.4f})")
axes[0].grid(True, alpha=0.3)

# Plot 2: 2D Contour (Dosage vs pH at center levels: Conc = 40 mg/L, Time = 80 min)
a_grid = np.linspace(0.1, 1.3, 80)
d_grid = np.linspace(2, 10, 80)
AA, DD = np.meshgrid(a_grid, d_grid)
BB = np.full_like(AA, 40.0)
CC = np.full_like(AA, 80.0)

X_grid = np.column_stack([
    np.ones(AA.size),
    AA.ravel(), BB.ravel(), CC.ravel(), DD.ravel(),
    (AA*BB).ravel(), (AA*CC).ravel(), (AA*DD).ravel(),
    (BB*CC).ravel(), (BB*DD).ravel(), (CC*DD).ravel(),
    (AA**2).ravel(), (BB**2).ravel(), (CC**2).ravel(), (DD**2).ravel()
])
ZZ = (X_grid @ beta).reshape(AA.shape)

cs = axes[1].contourf(AA, DD, ZZ, levels=14, cmap="viridis")
fig.colorbar(cs, ax=axes[1], label="Degradation (%)")
axes[1].set_xlabel("Catalyst Dosage (g/L)")
axes[1].set_ylabel("Initial pH")
axes[1].set_title("Contour Plot (Conc = 40 mg/L, Time = 80 min)")

plt.tight_layout()
plt.savefig("rsm_results.png", dpi=300)
plt.show()
