import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_openBF(dir_openBF):
    vases = ["vaso1", "vaso2", "vaso3"]
    variables = ["P", "Q", "A", "u"]

    # Criar dicionário para armazenar os dataframes
    data = {var: [] for var in variables}

    # Ler os arquivos e armazenar em dicionário
    for vase in vases:
        for var in variables:
            file_path = os.path.join(dir_openBF, f"{vase}_{var}.last")
            df = pd.read_csv(file_path, sep='\s+', header=None)
            df.columns = ["Time", "Length 1", "Length 2", "Length 3", "Length 4", "Length 5"]
            data[var].append(df)

    # Criar gráficos
    for var, label, unit in zip(["P", "Q", "A", "u"],
                                ["Pressure", "Flow", "Area", "Velocity"],
                                ["mmHg", "m³/s", "m²", "m/s"]):
        fig, axs = plt.subplots(3, 1, figsize=(10, 14))

        for i, vase in enumerate(vases):
            df = data[var][i]
            for j in range(1, 6):  # Colunas dos Nós (1 a 5)
                axs[i].scatter(df["Time"], df[f"Length {j}"], label=f"Length {j}")
                axs[i].plot(df["Time"], df[f"Length {j}"], linestyle='-', alpha=0.6)
            axs[i].set_title(f"{vase.capitalize()}: {label} vs Time")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel(f"{label} ({unit})")
            axs[i].grid(True)
            axs[i].legend()

        fig.subplots_adjust(hspace=0.7)
    plt.show()


# Exemplo de uso
plot_openBF("C:/Users/User/Documents/problema_inverso_results_openbf")

