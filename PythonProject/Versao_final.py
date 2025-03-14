import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from julia import Main


def update_openBF(base_file, update_file, output_file, openBF_dir):
    f"""
        Updates the .yaml file (with the description of the vessel network),
        simulates the vessels hemodynamics with openBF,
        plots the graphs with the output parameters and saves them automatically.

        Input:
        ----------

        base_file: str
            Path of the .yaml file to be updated.
        update_file: str
            Path of the .yaml file with the values that will overwrite the values of the base_file.
        output_file: str
            Path of the updated .yaml file.
            The .yaml file of the output_file does not need to exist previously, but the directory does.
            The inlet file needs to be in the same directory.
        openBF_dir: str
            Path of the directory where the .last files (openBF output) will be stored.
            The final folder of the directory does not need to exist previously.

        Return:
        -------

        .yaml updated file saved in {output_file}.
        openBF output files saved in {openBF_dir}.
        Plots saved in PNG, SVG, and PKL formats inside {openBF_dir}/plots.
        """

    def update_yaml(base_file: str, update_file: str, output_file: str = None):

        # Loads the base file
        with open(base_file, "r", encoding="utf-8") as f:
            yaml_base = yaml.safe_load(f) or {}

        # Loads the update file
        with open(update_file, "r", encoding="utf-8") as f:
            yaml_update = yaml.safe_load(f) or {}

        # Updates the base file values with those from the update file
        def overwrite(d1, d2):
            for key, value in d2.items(): # Loops through all keys and values of d2
                if isinstance(value, dict) and isinstance(d1.get(key), dict):
                    overwrite(d1[key], value) # Recursively calls the function if both values are dictionaries
                else:
                    d1[key] = value # If they are not dictionaries, overwrites the value of d1 with that of d2

        overwrite(yaml_base, yaml_update)

        # Saves the new .yaml file (output_file)
        output_file = output_file or base_file
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_base, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated .yaml file saved to: {output_file}")

    update_yaml(base_file, update_file, output_file)

    # Initializes the Julia environment
    jl = julia.Julia(compiled_modules=False)

    # Loads the required Julia packages
    Main.eval("using Pkg")

    # Loads the openBF package if it is not installed
    Main.eval(""" 
        if !haskey(Pkg.dependencies(), Base.UUID("e815b1a4-10eb-11ea-25f1-272ff651e618"))
        Pkg.add(url="https://github.com/INSIGNEO/openBF.git")
        end
    """)

    # Runs hemodynamic simulation
    Main.eval(""" 
        using openBF
        function pyopenBF(output_file, openBF_dir)
            run_simulation(output_file, verbose=true, save_stats=true, savedir=openBF_dir)
            println("openBF output saved in: $openBF_dir")
        end
    """)

    # Calls Julia function from Python
    Main.pyopenBF(output_file, openBF_dir)

    # Plots the simulation output graphs and saves them
    def plot_openBF(openBF_dir):
        vessels = ["vaso1", "vaso2", "vaso3"]
        variables = ["P", "Q", "A", "u"]
        titles = ["Vase 1", "Vase 2", "Vase 3"]

        # Creates dictionary to store dataframes
        data = {var: [] for var in variables}

        # Reads files and store in dictionary
        for vase in vessels:
            for var in variables:
                file_path = os.path.join(openBF_dir, f"{vase}_{var}.last")
                df = pd.read_csv(file_path, sep='\s+', header=None)
                df.columns = ["Time", "Length 1", "Length 2", "Length 3", "Length 4", "Length 5"]
                data[var].append(df)

                #data = {
                #   "P": [df_vaso1_P, df_vaso2_P, df_vaso3_P],
                #   "Q": [df_vaso1_Q, df_vaso2_Q, df_vaso3_Q],
                #   "A": [df_vaso1_A, df_vaso2_A, df_vaso3_A],
                #   "u": [df_vaso1_u, df_vaso2_u, df_vaso3_u]
                #}

        # Creates folder for saving plots
        plots_dir = os.path.join(openBF_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plots graphs
        for var, label, unit in zip(["P", "Q", "A", "u"],
                                    ["Pressure", "Flow", "Area", "Velocity"],
                                    ["mmHg", "m³/s", "m²", "m/s"]):
            fig, axs = plt.subplots(3, 1, figsize=(10, 14))

            for i, title in enumerate(titles):
                df = data[var][i]
                for j in range(1, 6):  # Colunas dos Nós (1 a 5)
                    axs[i].scatter(df["Time"], df[f"Length {j}"], label=f"Length {j}")
                    axs[i].plot(df["Time"], df[f"Length {j}"], linestyle='-', alpha=0.6)
                axs[i].set_title(f"{title.capitalize()}: {label} vs Time")
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel(f"{label} ({unit})")
                axs[i].grid(True)
                axs[i].legend()

            fig.subplots_adjust(hspace=0.7)

            # Saves plots in .png .svg and .pdf formats
            plot_path = os.path.join(plots_dir, f"{var}_plot")

            plt.savefig(f"{plot_path}.png", dpi=300)
            plt.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)

            print(f"Plots saved: {plot_path}.png, {plot_path}.svg, {plot_path}.pkl")

        plt.show()

    plot_openBF(openBF_dir)



if __name__ =="__main__":


    # Applying the function with my files

    base_file = "C:/Users/User/OneDrive/Documentos/BIBI/Mestrado EBM/Simulação openBF/problema_inverso_Automatizado.yaml"
    update_file = "C:/Users/User/OneDrive/Documentos/BIBI/Mestrado EBM/Simulação openBF/entradas.yaml"
    output_file = "C:/Users/User/Documents/problema_inverso_results_pycharm/resultado.yaml"
    openBF_dir = "C:/Users/User/Documents/problema_inverso_results_openbf"

    update_openBF(base_file, update_file, output_file, openBF_dir)
