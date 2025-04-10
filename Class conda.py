import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

from julia import Main

class OPENBF_Jacobian:
    """
            Creates the Jacobian matrix for a parameter variation of a vessel.

            Attributes:
            ----------

            base_file: str
                Path of the .yaml file to be updated.
            updated_file: str
                Path of the updated .yaml file.
                The .yaml file of the output_file does not need to exist previously, but the directory does.
                The inlet file needs to be in the same directory.
            openBF_dir: str
                Path of the directory where the .last files (openBF output) will be stored.
                The final folder of the directory does not need to exist previously.
            """
    def __init__(self, base_file, updated_file, openBF_dir):
        self.base_file = base_file
        self.updated_file = updated_file
        self.openBF_dir = openBF_dir


    def update_yaml(self, vase, parameter, add_value):

        self.add_value = add_value

        # Loads YAML from base_file
        with open(self.base_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the YAML.")
            return

        found = False  # Flag to know if the vase was found

        for item in yaml_data["network"]:
            print(f" Checking vessel: {item.get('label')}")  # Debug print
            if item.get("label") == vase:
                if parameter in item:
                    item[parameter] = item[parameter] + add_value
                    print(f"Parameter '{parameter}' of vessel '{vase}' updated to: {item[parameter]}")
                    found = True
                else:
                    print(f"Error: Parameter '{parameter}' not found in vessel '{vase}'.")
                break

        if not found:
            print(f"Error: Vessel '{vase}' not found in YAML.")

        # Saves the updated YAML to the updated_file
        with open(self.updated_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        print(f"Updated file saved in: {self.updated_file}")


    def openBF(self, file, output_dir):
        """Runs openBF"""
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
                                    function pyopenBF(file, output_dir)
                                        run_simulation(file, verbose=true, save_stats=true, savedir=output_dir)
                                        println("openBF output saved in: $output_dir")
                                    end
                                """)

        # Calls Julia function from Python
        Main.pyopenBF(file, output_dir)
        return

    def stack_last_files(self, vessel, variables, data_dir):
        """Stacks the openBF output for each vessel in the order "A", "P", "Q" and "u" from top to bottom
        and saves it to a .last file"""
        data_list = []

        for var in variables:
            file_path = os.path.join(data_dir, f"{vessel}_{var}.last")

            # Checks if file exists
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found.")
                return

            # Reads data from the .last file
            data = np.loadtxt(file_path)
            data_list.append(data)

        # Stacks files vertically (A, P, Q, u)
        stacked_data = np.vstack(data_list)

        # Saves the stacked file
        stacked_file = os.path.join(data_dir, f"{vessel}_stacked.last")
        np.savetxt(stacked_file, stacked_data, fmt="%.14e")  # 6 decimal places

        print(f"Saved file: {stacked_file}")

    def partial_deriv_files(self, base_dir, updated_dir, del_dir, parameter):
        """
        Subtracts the stacked files of base_dir from the stacked files of updated_dir,
        divides by the delta parameter and saves the result in del_dir.
        """
        vessels = ["vase1", "vase2", "vase3"]


        # Calculates the parameter difference
        delta_value = self.add_value
        print(f"Parameter '{parameter}' difference: {delta_value}")

        if delta_value == 0:
            print("Warning: Parameter difference is zero. Avoiding division by zero.")
            return

        # Creates the output directory if it does not exist
        os.makedirs(del_dir, exist_ok=True)

        for vessel in vessels:
            base_file_path = os.path.join(base_dir, f"{vessel}_stacked.last")
            updated_file_path = os.path.join(updated_dir, f"{vessel}_stacked.last")
            del_file_path = os.path.join(del_dir, f"{vessel}_del_{parameter}_delta={delta_value}.last")

            # Checks if both files exist
            if not os.path.exists(base_file_path) or not os.path.exists(updated_file_path):
                print(f"Error: Files for {vessel} not found. Skipped.")
                continue

            # Loads the files .last
            base_data = np.loadtxt(base_file_path)
            updated_data = np.loadtxt(updated_file_path)

            # Checks if the dimensions are compatible
            if base_data.shape != updated_data.shape:
                print(f"Error: Incompatible dimensions for {vessel}: {base_data.shape} vs {updated_data.shape}.")
                continue

            # Performs subtraction and division by delta_value
            del_data = (updated_data - base_data) / delta_value

            # Saves the result
            np.savetxt(del_file_path, del_data, fmt="%.14e")
            print(f"Partial derivatives file saved: {del_file_path}")


    def plot_openBF(self, data_dir):
        """ Plots the openBF output graphs (Pressure/Area/Flow/Velocity vs. Time)
        and saves them in data_dir.
        """
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

    def stack_partial_derivatives(self, delta_dict, output_path):
        """
        Horizontally stacks the partial derivative matrices for each vessel.

        Parameters:
        delta_dict (dict): Dictionary with the deltas used in each parameter.
        output_path (str): Path where the stacked files will be saved.
        """
        parameters = ["E", "h0", "L", "R0"]
        vessels = ["vase1", "vase2", "vase3"]

        # Filtra os parâmetros com delta != 0
        valid_parameters = []
        for param in parameters:
            if delta_dict[param] != 0:
                valid_parameters.append(param)
            else:
                print(
                    f"Warning: Variation of the parameter '{param}' is zero. It will be excluded from the Jacobian matrix.")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for vessel in vessels:
            matrices = []

            for param in valid_parameters:
                delta = delta_dict[param]
                file_path = os.path.join(openBF_dir, f"partial_deriv_{param}",
                                         f"{vessel}_del_{param}_delta={delta}.last")

                if os.path.exists(file_path):
                    matrix = np.loadtxt(file_path)
                    matrices.append(matrix)
                else:
                    print(f"Error: File not found - {file_path}.")
                    return

            if matrices:
                stacked_matrix = np.hstack(matrices)
                output_file = os.path.join(output_path, f"jacobian_{vessel}_stacked.txt")
                np.savetxt(output_file, stacked_matrix, fmt="%.14e")
                print(f"Stacked matrix saved in: {output_file}")

    def pseudoinverse_matrix(self):
        vessels = ["vase1", "vase2", "vase3"]

        for vessel in vessels:
            file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_{vessel}_stacked.txt")

            if os.path.exists(file_path):
                J0 = np.loadtxt(file_path) # Loads the Jacobian matrix
                J0T_J0 = J0.T @ J0 # Jacobian transpose times the Jacobian

                # Checks if it is invertible
                if np.linalg.det(J0T_J0) == 0:
                    print("Error: Non-invertible matrix.")
                    return

                # Calculates the pseudoinverse matrix
                J0T_J0_inv = np.linalg.inv(J0T_J0)
                pseudoinv = J0T_J0_inv @ J0.T

                output_dir = os.path.join(openBF_dir, "jacobians_pseudoinverse")
                os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it does not exist

                output_file = os.path.join(output_dir, f"pseudo_inv_{vessel}.txt")
                np.savetxt(output_file, pseudoinv, fmt="%.14e")
                print(f"Pseudoinverse matrix saved in: {output_file}")

            else:
                print(f"Error: File not found - {file_path}.")
                return

    def base_openBF(self):
        """Runs openBF in Julia for the base YAML;
            stacks the output values;
            and plots the graphs (Pressure/Area/Flow/Velocity vs. Time)."""

        # Where the base_file output files will be
        base_dir = os.path.join(openBF_dir, "openBF_base")
        os.makedirs(base_dir, exist_ok=True)

        # Runs openBF to base_file
        self.openBF(base_file,base_dir)

        # Stacking order of vessels and variables
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["A", "P", "Q", "u"]

        # Stack openBF outputs for each vessel individually
        for vessel in vessels:
            self.stack_last_files(vessel, variables, base_dir)

        # Plots the simulation output graphs and saves them
        self.plot_openBF(base_dir)


    def updated_openBF(self, vase, parameter, add_value):
        """Updates the value of a parameter within a specific vessel;
        runs openBF in Julia for the updated YAML;
        stacks the output values, calculates the partial derivatives with respect to the modified parameter;
        and plots the graphs (Pressure/Area/Flow/Velocity vs. Time)."""


        self.update_yaml(vase, parameter, add_value)

        # Where the base_file output files are
        base_dir = os.path.join(openBF_dir, "openBF_base")
        # Where the updated_file output files will be
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_{vase}_{parameter}")
        os.makedirs(updated_dir, exist_ok=True)

        # Runs openBF to updated_file
        self.openBF(updated_file,updated_dir)

        # Stacking order of vessels and variables
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["A", "P", "Q", "u"]

        # Stack openBF outputs for each vessel individually
        for vessel in vessels:
            self.stack_last_files(vessel, variables, updated_dir)

        # Path to the partial derivatives directory
        del_dir = os.path.join(openBF_dir, f"partial_deriv_{parameter}")

        # Calculates and creates the partial derivatives files
        self.partial_deriv_files(base_dir, updated_dir, del_dir, parameter)

        # Plots the simulation output graphs and saves them
        self.plot_openBF(updated_dir)

    def jacobian_pseudoinv_matrix(self, vase, add_E, add_h0, add_L, add_R0):
        """Creates the Jacobian pseudoinverse matrix considering the increments specified for each parameter."""
        add_values = {"E": add_E, "h0": add_h0, "L": add_L, "R0": add_R0}

        for parameter in add_values:
            self.updated_openBF(vase, parameter, add_values[parameter])

        # Path to the Jacobian matrices directory
        output_path = os.path.join(openBF_dir, f"jacobians")
        self.stack_partial_derivatives(add_values, output_path)

        # Creates the pseudoinverse matrix
        self.pseudoinverse_matrix()

    def optimized_parameters(self, vase, add_E, add_h0, add_L, add_R0):
        """ Calculates the optimized parameters by multiplying the pseudoinverse Jacobian with the stacked output matrix
        for each vessel where the variation is non-zero.
        Saves the results in the folder 'optimized_parameters'."""

        vessels = ["vase1", "vase2", "vase3"]
        parameters = ["E", "h0", "L", "R0"]
        add_values = {"E": add_E, "h0": add_h0, "L": add_L, "R0": add_R0}

        output_dir = os.path.join(self.openBF_dir, "optimized_parameters")
        os.makedirs(output_dir, exist_ok=True)

        for vessel in vessels:
            # Verifica se existe pelo menos um parâmetro modificado
            has_nonzero_param = any(add_values[param] != 0 for param in parameters)
            if not has_nonzero_param:
                print(f"Skipping {vessel} (no parameter variation).")
                continue

            pseudo_path = os.path.join(self.openBF_dir, "jacobians_pseudoinverse", f"pseudo_inv_{vessel}.txt")
            if not os.path.exists(pseudo_path):
                print(f"Error: Pseudoinverse not found for {vessel} - {pseudo_path}")
                continue

            # Carrega a pseudoinversa
            pseudo_inv = np.loadtxt(pseudo_path)

            # Acha o parâmetro que teve valor alterado (pode ser mais de um)
            for param in parameters:
                if add_values[param] != 0:
                    stacked_path = os.path.join(self.openBF_dir, f"openBF_updated_{vase}_{param}",
                                                f"{vessel}_stacked.last")
                    if os.path.exists(stacked_path):
                        stacked_output = np.loadtxt(stacked_path)
                        break  # pega o primeiro encontrado com add_value ≠ 0
            else:
                print(f"Warning: No stacked output found for {vessel}. Skipping.")
                continue

            # Verifica dimensão
            if pseudo_inv.shape[1] != stacked_output.shape[0]:
                print(
                    f"Error: Shape mismatch for {vessel} - pseudo_inv: {pseudo_inv.shape}, output: {stacked_output.shape}")
                continue

            # Multiplica pseudoinversa pela saída empilhada
            x_opt = pseudo_inv @ stacked_output

            output_file = os.path.join(output_dir, f"optimized_{vessel}.txt")
            np.savetxt(output_file, x_opt, fmt="%.14e")
            print(f"Optimized parameters for {vessel} saved in: {output_file}")


# Application
if __name__ == "__main__":

    base_file = "C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso - Alvo.yaml"
    updated_file = "C:/Users/User/Documents/problema_inverso_results_openbf/resultado.yaml"
    openBF_dir = "C:/Users/User/Documents/problema_inverso_results_openbf"

    updater = OPENBF_Jacobian(base_file, updated_file, openBF_dir)

    # Runs openBF to base_file
    #updater.base_openBF()

    # Creates the Jacobian pseudoiniverse matrix considering the increments specified for each parameter
    #updater.jacobian_pseudoinv_matrix("vase1", 0, 0.001, 0, 0)

    # Creates the optimized output using the pseudoinverse matrix
    updater.optimized_parameters("vase1", 0, 0.001, 0, 0)


