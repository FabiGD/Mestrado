import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

from julia import Main

class OPENBF_Jacobean:
    def __init__(self, base_file, updated_file, openBF_dir):
        self.base_file = base_file
        self.updated_file = updated_file
        self.openBF_dir = openBF_dir

    def load_yaml(self):
        """Loads YAML from base_file"""
        with open(self.base_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save_yaml(self, data):
        """Saves the updated YAML to the updated_file"""
        with open(self.updated_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"Updated file saved in: {self.updated_file}")

    def update_yaml(self, vase, parameter, new_value):
        """Updates the value of a parameter within a specific vessel;
        runs openBF in Julia for both the base YAML and the updated YAML;
        stacks the output values, calculates the partial derivatives with respect to the modified parameter;
        and plots the graphs (Pressure/Area/Flow/Velocity vs. Time)."""

        yaml_data = self.load_yaml()

        if "network" not in yaml_data:
            print("The 'network' key was not found in the YAML.")
            return

        found = False  # Flag to know if the vase was found

        for item in yaml_data["network"]:
            print(f" Checking vessel: {item.get('label')}")  # Debug print
            if item.get("label") == vase:
                if parameter in item:
                    item[parameter] = new_value
                    print(f"Parameter '{parameter}' of vessel '{vase}' updated to: {new_value}")
                    found = True
                else:
                    print(f"Parameter '{parameter}' not found in vessel '{vase}'.")
                break

        if not found:
            print(f"Vessel '{vase}' not found in YAML.")

        self.save_yaml(yaml_data)

        # Creates folder for saving plots
        # Where the base_file output files will be
        base_dir = os.path.join(openBF_dir, "openBF_base")
        os.makedirs(base_dir, exist_ok=True)

        # Where the updated_file output files will be
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_{vase}_{parameter}={new_value}")
        os.makedirs(updated_dir, exist_ok=True)

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

        # Runs hemodynamic simulation for base_file
        Main.eval(""" 
                    using openBF
                    function pyopenBF(base_file, base_dir)
                        run_simulation(base_file, verbose=true, save_stats=true, savedir=base_dir)
                        println("openBF output saved in: $base_dir")
                    end
                """)

        # Calls Julia function from Python
        Main.pyopenBF(base_file, base_dir)

        # Runs hemodynamic simulation for updated_file
        Main.eval(""" 
            using openBF
            function pyopenBF(updated_file, updated_dir)
                run_simulation(updated_file, verbose=true, save_stats=true, savedir=updated_dir)
                println("openBF output saved in: $updated_dir")
            end
        """)

        # Calls Julia function from Python
        Main.pyopenBF(updated_file, updated_dir)

        # Stacking order of vessels and variables
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["A", "P", "Q", "u"]

        def stack_last_files(vessel, variables, data_dir):
            """Stacks the openBF output for each vessel in the order "A", "P", "Q" and "u" from top to bottom
            and saves it to a .last file"""
            data_list = []

            for var in variables:
                file_path = os.path.join(data_dir, f"{vessel}_{var}.last")

                # Checks if file exists
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found.")
                    return

                # Reads data from the .last file
                data = np.loadtxt(file_path)
                data_list.append(data)

            # Stacks files vertically (A, P, Q, u)
            stacked_data = np.vstack(data_list)

            # Saves the stacked file
            updated_file = os.path.join(data_dir, f"{vessel}_stacked.last")
            np.savetxt(updated_file, stacked_data, fmt="%.6e") # 6 decimal places

            print(f"Saved file: {updated_file}")

        # Processes each vessel individually
        for vessel in vessels:
            stack_last_files(vessel, variables, base_dir)
            stack_last_files(vessel, variables, updated_dir)

        def partial_deriv_files(base_dir, updated_dir, del_dir, base_file, updated_file, vase, parameter):
            """
            Subtracts the stacked files of base_dir from the stacked files of updated_dir,
            divides by the delta parameter and saves the result in del_dir.
            """
            vessels = ["vase1", "vase2", "vase3"]

            def load_param_value(yaml_file, vase, parameter):
                """Loads the specified parameter value of a specified vessel."""
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}

                for item in yaml_data.get("network", []):
                    if item.get("label") == vase:
                        return item.get(parameter)

                print(f"Parameter '{parameter}' not found for vessel '{vase}'.")
                return None

            # Gets the parameter values in YAML files
            value_base = load_param_value(base_file, vase, parameter)
            value_updated = load_param_value(updated_file, vase, parameter)

            if value_base is None or value_updated is None:
                print("Could not get parameter values. Aborting calculation.")
                return

            # Calculates the parameter difference
            delta_value = value_updated - value_base
            print(f"Parameter '{parameter}' difference: {value_updated} - {value_base} = {delta_value}")

            if delta_value == 0:
                print("Parameter difference is zero. Avoiding division by zero.")
                return

            # Creates the output directory if it does not exist
            os.makedirs(del_dir, exist_ok=True)

            for vessel in vessels:
                base_file_path = os.path.join(base_dir, f"{vessel}_stacked.last")
                updated_file_path = os.path.join(updated_dir, f"{vessel}_stacked.last")
                del_file_path = os.path.join(del_dir, f"{vessel}_del_{parameter}_delta={delta_value}.last")

                # Checks if both files exist
                if not os.path.exists(base_file_path) or not os.path.exists(updated_file_path):
                    print(f"Files for {vessel} not found. Skipped.")
                    continue

                # Loads the files .last
                base_data = np.loadtxt(base_file_path)
                updated_data = np.loadtxt(updated_file_path)

                # Checks if the dimensions are compatible
                if base_data.shape != updated_data.shape:
                    print(f"Incompatible dimensions for {vessel}: {base_data.shape} vs {updated_data.shape}")
                    continue

                # Performs subtraction and division by delta_value
                del_data = (updated_data - base_data) / delta_value

                # Saves the result
                np.savetxt(del_file_path, del_data, fmt="%.6e")
                print(f"Partial derivatives file saved: {del_file_path}")

        # Path to the partial derivatives directory
        del_dir = os.path.join(openBF_dir, f"partial_deriv_{parameter}")

        # Calls the function
        partial_deriv_files(base_dir, updated_dir, del_dir, base_file, updated_file, vase, parameter)

        # Plots the simulation output graphs and saves them
        def plot_openBF(data_dir):
            vessels = ["vase1", "vase2", "vase3"]
            variables = ["P", "Q", "A", "u"]
            titles = ["Vase 1", "Vase 2", "Vase 3"]

            # Creates dictionary to store dataframes
            data = {var: [] for var in variables}

            # Reads files and store in dictionary
            for vase in vessels:
                for var in variables:
                    file_path = os.path.join(data_dir, f"{vase}_{var}.last")
                    df = pd.read_csv(file_path, sep='\s+', header=None)
                    df.columns = ["Time", "Length 1", "Length 2", "Length 3", "Length 4", "Length 5"]
                    data[var].append(df)

                    # data = {
                    #   "P": [df_vase1_P, df_vase2_P, df_vase3_P],
                    #   "Q": [df_vase1_Q, df_vase2_Q, df_vase3_Q],
                    #   "A": [df_vase1_A, df_vase2_A, df_vase3_A],
                    #   "u": [df_vase1_u, df_vase2_u, df_vase3_u]
                    # }

            # Creates folder for saving plots
            plots_dir = os.path.join(data_dir, "plots")
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

        plot_openBF(base_dir)
        plot_openBF(updated_dir)


# Application
if __name__ == "__main__":

    base_file = "C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso_Automatizado.yaml"
    updated_file = "C:/Users/User/Documents/problema_inverso_results_openbf/resultado.yaml"
    openBF_dir = "C:/Users/User/Documents/problema_inverso_results_openbf"

    updater = OPENBF_Jacobean(base_file, updated_file, openBF_dir)

    # Update vessel X in parameter Y to a new value Z
    updater.update_yaml("vase1", "R0", 0.026)
