import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time

from julia import Main
from matplotlib.ticker import ScalarFormatter

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
    def __init__(self, patient_file, k0_file, openBF_dir):
        self.patient_file = patient_file
        self.k0_file = k0_file # Is not used, but needs to exist
        self.openBF_dir = openBF_dir


    def update_yaml(self, knumber, vase, parameter, add_value):

        self.add_value = add_value
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}.yaml")

        # Loads YAML from k_file
        if knumber == 0:
            k_file = os.path.join(openBF_dir, self.k0_file)
        else:
            k_file = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")
        with open(k_file, "r", encoding="utf-8") as f:
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
        with open(updated_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        print(f"Updated file saved in: {updated_file}")


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

        # Runs hemodynamic simulation
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
        """Stacks the openBF output for each vessel in the order of the variables from top to bottom
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
        np.savetxt(stacked_file, stacked_data, fmt="%.14e")

        print(f"Saved file: {stacked_file}")

    def partial_deriv_files(self, vessel, base_dir, updated_dir, del_dir, parameter):
        """
        Subtracts the stacked files of base_dir from the stacked files of updated_dir,
        divides by the delta parameter and saves the result in del_dir.
        """

        # Calculates the parameter difference
        delta_value = self.add_value
        print(f"Parameter '{parameter}' difference: {delta_value}")

        if delta_value == 0:
            print("Warning: Parameter difference is zero. Avoiding division by zero.")
            return

        # Creates the output directory if it does not exist
        os.makedirs(del_dir, exist_ok=True)

        base_file_path = os.path.join(base_dir, f"{vessel}_stacked.last") #!!!
        updated_file_path = os.path.join(updated_dir, f"{vessel}_stacked.last")
        del_file_path = os.path.join(del_dir, f"{vessel}_del_{parameter}_delta={delta_value}.last")

        # Checks if both files exist
        if not os.path.exists(base_file_path) or not os.path.exists(updated_file_path):
            print(f"Error: Files for {vessel} not found. Skipped.")
            return

        # Loads the files .last
        base_data = np.loadtxt(base_file_path)
        updated_data = np.loadtxt(updated_file_path)

        # Checks if the dimensions are compatible
        if base_data.shape != updated_data.shape:
            print(f"Error: Incompatible dimensions for {vessel}: {base_data.shape} vs {updated_data.shape}.")

        # Performs subtraction and division by delta_value
        del_data = (updated_data - base_data) / delta_value

        # Saves the result
        np.savetxt(del_file_path, del_data, fmt="%.14e")
        print(f"Partial derivatives file saved: {del_file_path}")


    def plot_openBF(self, vessel, data_dir):
        """ Plots the openBF output graphs (Pressure/Area/Flow/Velocity vs. Time)
        and saves them in data_dir.
        """

        variables = ["P", "u"]

        # Creates dictionary to store dataframes
        data = {var: [] for var in variables}

        # Reads files and store in dictionary
        for var in variables:
            file_path = os.path.join(data_dir, f"{vessel}_{var}.last")
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            df.columns = ["Time", "Length 1", "Length 2", "Length 3", "Length 4", "Length 5"]
            data[var].append(df)

            # data = {
            #   "P": [df_vase1_P, df_vase2_P, df_vase3_P],
            #   "u": [df_vase1_u, df_vase2_u, df_vase3_u]
            # }

        # Creates folder for saving plots
        plots_dir = os.path.join(data_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plots graphs
        for var, label, unit in zip(["P", "u"],
                                    ["Pressure", "Velocity"],
                                    ["mmHg", "m/s"]):
            fig, axs = plt.subplots(3, 1, figsize=(10, 14))

            df = data[var][0]
            for j in range(1, 6):  # Colunas dos Nós (1 a 5)
                axs.scatter(df["Time"], df[f"Length {j}"], label=f"Length {j}")
                axs.plot(df["Time"], df[f"Length {j}"], linestyle='-', alpha=0.6)
            axs.set_title(f"{vessel.capitalize()}: {label} vs Time")
            axs.set_xlabel("Time (s)")
            axs.set_ylabel(f"{label} ({unit})")
            axs.grid(True)
            axs.legend()

            fig.subplots_adjust(hspace=0.7)

            # Saves plots in .png .svg and .pdf formats
            plot_path = os.path.join(plots_dir, f"{var}_plot")

            plt.savefig(f"{plot_path}.png", dpi=300)
            plt.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)

            plt.close(fig)

            print(f"Plots saved: {plot_path}.png, {plot_path}.svg, {plot_path}.pkl")

    def stack_partial_derivatives(self, vessel, delta_dict, output_path):
        """
        Horizontally stacks only the 4th column of the partial derivative matrices for each vessel.

        Parameters:
        delta_dict (dict): Dictionary with the deltas used in each parameter.
        output_path (str): Path where the stacked files will be saved.
        """
        parameters = ["h0", "L", "R0", "Rp", "Rd", "E"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        for param in parameters:
            if delta_dict[param] == 0:
                print(
                    f"Warning: Variation of the parameter '{param}' is zero. It will be excluded from the Jacobian matrix.")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        matrices = []

        for param in valid_parameters:
            delta = delta_dict[param]
            file_path = os.path.join(openBF_dir, f"partial_deriv_{param}",
                                        f"{vessel}_del_{param}_delta={delta}.last")

            if os.path.exists(file_path):
                matrix = np.loadtxt(file_path)
                fourth_column = matrix[:, 3].reshape(-1, 1)  # Selects the 4th column and keeps 2D format
                matrices.append(fourth_column)
            else:
                print(f"Error: File not found - {file_path}.")
                return

        if matrices:
            stacked_matrix = np.column_stack(matrices)
            output_file = os.path.join(output_path, f"jacobian_{vessel}_stacked.txt")
            np.savetxt(output_file, stacked_matrix, fmt="%.14e")
            print(f"Stacked matrix saved in: {output_file}")

    def M_matrix(self, beta, vessel, knumber, valid_parameters):
        """ Creates the M_matrix using the Jacobian matrix and saves it in the folder: "M_matrix_beta={beta}"."""

        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_{vessel}_stacked.txt")
        print_path = os.path.join(openBF_dir, f"M_matrix_beta={beta}", f"condition_{vessel}_k{knumber}.txt")

        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
            JkT_Jk = Jk.T @ Jk # Jacobian transpose times the Jacobian

            # Define the regularization weights for each parameter
            weights = {
                "H0": 1.0,
                "L": 1e-4,
                "R0": 1e-2,
                "Rp": 1e-2,
                "Rd": 1e-2,
                "E": 1e-14
            }

            # Creates the diagonal matrix D based on the valid_parameters order
            D_values = [weights.get(param, 1.0) for param in valid_parameters]
            D = np.diag(D_values)
            print("Diagonal matrix D:")
            print(D)

            # Regularizing the condition number of the inverse matrix
            #beta = 1e1 # apagar: a partir de 1e12 (1e14 para beta = 1e-3 * np.linalg.norm(JkT_Jk))
            Jk_beta = JkT_Jk + beta * D 

            # Checks if it is invertible
            # Calculates the condition number
            cond_Jk = np.linalg.cond(Jk)
            cond_matrix = np.linalg.cond(JkT_Jk)
            cond_beta = np.linalg.cond(Jk_beta)

            # Sets a threshold to consider "non-invertible"
            threshold = 1e6

            # Checks the condition number of matrices
            matrices = [
                (cond_Jk, f"Jk matrix"),
                (cond_matrix, f"JkT@Jk matrix"),
                (cond_beta, f"(JkT@Jk + beta * D) matrix")
            ]

            # Prints and saves in a file the condition number and the status of the matrices
            os.makedirs(os.path.dirname(print_path), exist_ok=True)

            with open(print_path, "w") as log:
                for cond, desc in matrices:
                    msg_cond = f"{desc} condition number: {cond:.2e}"
                    msg_status = (
                        f"The {desc} is invertible."
                        if cond < threshold
                        else f"Error: The {desc} is quasi-singular or non-invertible."
                    )
                    print(msg_cond)
                    print(msg_status)
                    log.write(msg_cond + "\n")
                    log.write(msg_status + "\n")
                    log.write("\n")
                print(f"beta = {beta}")
                log.write(f"beta = {beta} \n")

            # Saves matrix Jk_beta (M matrix)
            output_dir = os.path.join(openBF_dir, f"M_matrix_beta={beta}")
            os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it does not exist

            output_file = os.path.join(output_dir, f"Jk_beta_{vessel}.txt")
            np.savetxt(output_file, Jk_beta, fmt="%.14e")
            print(f"Jk_beta (M) matrix saved in: {output_file}")

        else:
            print(f"Error: File not found - {file_path}.")
            return

    def Z_matrix(self, vessel, knumber):

        # Path to the jacobian matrix file
        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_{vessel}_stacked.txt")

        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
        else:
            raise SystemExit(f"Error: Jacobian matrix file not found - {file_path}")

        # Loads the data from the patient openBF output - ym
        patient_output = os.path.join(openBF_dir, f"ym - openBF output paciente", f"{vessel}_stacked.last")

        if not os.path.exists(patient_output):
            raise SystemExit(f"Error: Patient output file not found - {patient_output}")

        data = np.loadtxt(patient_output)
        patient_data = data[:, 3].reshape(-1, 1) # Only takes the 3rd knot

        # Loads the simulation output corresponding to the guess
        yk_output = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}", f"{vessel}_stacked.last")

        if not os.path.exists(yk_output):
            raise SystemExit(f"Error: Output file for iteration {knumber} not found - {yk_output}")

        data = np.loadtxt(yk_output)
        yk_data = data[:, 3].reshape(-1, 1)  # Only takes the 3rd knot

        y_tilde = patient_data - yk_data

        # Creates the Z matriz
        Z = Jk.T @ y_tilde

        # Where the Z matrix files will be
        Z_dir = os.path.join(openBF_dir, "Z_matrix")
        os.makedirs(Z_dir, exist_ok=True)

        # Saves the y_tilde matrix in a file
        Z_file = os.path.join(Z_dir, f"Z_matrix_{vessel}.last")
        np.savetxt(Z_file, Z, fmt="%.14e")
        print(f"Z matrix saved: {Z_file}")

    def Pdk(self, vessel, delta_dict, param_directory, yaml_file):
        """Loads the parameters of a yaml file and saves it in a directory."""

        Pdk = {vessel: []}

        # Creates a vector with the parameter values corresponding to the guess
        parameters = ["h0", "L", "R0", "Rp", "Rd", "E"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        # Loads YAML from k_file
        k_file = os.path.join(openBF_dir, yaml_file)
        with open(k_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the YAML.")
            return

        found = False  # Flag to know if the vase was found

        for item in yaml_data["network"]:
            if item.get("label") == vessel:
                for parameter in valid_parameters:
                    if parameter in item:
                        value = item[parameter]
                        Pdk[vessel].append(value)
                        found = True
                    else:
                        print(f"Error: Parameter '{parameter}' not found in vessel '{vessel}'.")
                        return

        if not found:
            print(f"Error: Vessel '{vessel}' not found in YAML.")

        # Where the parameters files will be
        file_dir = os.path.join(openBF_dir, param_directory)
        os.makedirs(file_dir, exist_ok=True)

        # Saves the parameters vector in a file
        Pdk_file = os.path.join(file_dir, f"Pdk_{vessel}.last")
        param_array = np.array(Pdk[vessel]).reshape(-1, 1)
        np.savetxt(Pdk_file, param_array, fmt="%.14e")
        print(f"Parameter vector saved: {Pdk_file}")


    def optimized_parameters(self, vessel, alpha, beta, knumber):
        """ Obtains the optimized parameters Pdk+1 and saves them in a file.
        
        Parameters:
            alpha (int): Alpha is the sub-relaxation factor for the Newton step."""

        # Where the matrix of optimized parameters will be
        new_knumber = knumber + 1
        opt_param_dir = os.path.join(openBF_dir, f"optimized_parameters_Pd{new_knumber}")
        os.makedirs(opt_param_dir, exist_ok=True)

        # Loads the parameters of the initial guess (Pdk)
        if knumber == 0:
            param_path = os.path.join(openBF_dir, f"Pd{knumber}", f"Pdk_{vessel}.last")
        else:
            param_path = os.path.join(openBF_dir, f"optimized_parameters_Pd{knumber}",
                                        f"Pdk_{vessel}.last")

        if os.path.exists(param_path):
            param_data = np.loadtxt(param_path).reshape(-1, 1)
        else:
            print(f"Error: Pdk matrix file not found - {param_path}.")
            return

        # Loads the data from the M matrix
        M_matrix_path = os.path.join(openBF_dir, f"M_matrix_beta={beta}", f"Jk_beta_{vessel}.txt")

        if os.path.exists(M_matrix_path):
            M_data = np.loadtxt(M_matrix_path)
        else:
            print(f"Error: Pseudoinverse matrix file not found - {M_matrix_path}.")
            return

        # Loads the data from the Z matrix
        Z_matrix_path = os.path.join(openBF_dir, "Z_matrix", f"Z_matrix_{vessel}.last")

        if os.path.exists(Z_matrix_path):
            Z_data = np.loadtxt(Z_matrix_path).reshape(-1, 1)

        else:
            print(f"Error: Z matrix file not found - {Z_matrix_path}.")
            return

        # Creates the optimized parameters (Pd(k+1)) matrix
        P_matrix = np.linalg.solve(M_data,Z_data)
        opt_param_data = param_data + alpha * P_matrix

        # Saves the optimized parameters matrix in a file
        opt_param_file = os.path.join(opt_param_dir, f"Pdk_{vessel}.last")
        np.savetxt(opt_param_file, opt_param_data, fmt="%.14e")
        print(f"Optimized parameters matrix saved: {opt_param_file}")

        # Checking optimized parameters: prevents negative values
        if np.any(opt_param_data < 0):
            raise ValueError(
                f"Invalid parameter: Negative value detected in {opt_param_data.flatten()}.\n"
                f"Check the value of alpha or the input data."
            )


    def update_yaml_with_optimized_parameters(self, vessel, delta_dict, base_yaml_path, param_files_dir, output_yaml_path):
        # Updates the input YAML using the optimized parameters saved in separate files.

        parameters = ["h0", "L", "R0", "Rp", "Rd", "E"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        # Loads the YAML file
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: 'network' key not found in YAML.")
            return

        # Loads the file with the optimized parameters
        param_file = os.path.join(param_files_dir, f"Pdk_{vessel}.last")

        if os.path.exists(param_file):
            new_params = np.loadtxt(param_file)
        else:
            print(f"Error: Parameter file not found - {param_file}. Skipping {vessel}.")
            return

        # Ensures that new_params is a vector (not an array)
        new_params = np.atleast_1d(new_params)

        if len(new_params) != len(valid_parameters):
            print(
                f"Error: Number of parameters mismatch for {vessel}. Expected {len(valid_parameters)}, got {len(new_params)}.")

        # Updates the YAML values
        for item in yaml_data["network"]:
            if item.get("label") == vessel:
                for i, param in enumerate(valid_parameters):
                    item[param] = float(new_params[i])
                print(f"Updated parameters for {vessel}: {new_params}")
                break
        else:
            print(f"Warning: Vessel {vessel} not found in YAML.")

        # Saves the new YAML file
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated YAML saved in: {output_yaml_path}")

    def plot_RMSE(self, vessel, beta, data_dir, knumber_max):
        # Plots the average RMSE vs. iteration

        plt.close('all')
        rmse_means = []

        for knumber in range(0, knumber_max + 1):

            rmse_totals_k = []  # List to store the total RMSE of the vessel

            # Files paths
            patient_file = os.path.join(openBF_dir, "ym - openBF output paciente", f"{vessel}_stacked.last")
            k_file = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}", f"{vessel}_stacked.last")

            # Checks if files exist
            if not os.path.exists(patient_file):
                print(f"Error: File {patient_file} not found.")
                return
            if not os.path.exists(k_file):
                print(f"Error: File {k_file} not found.")
                return


            # Loads stacked files ignoring comments, takes only the 4ª column
            patient_data = np.loadtxt(patient_file, comments="#")[:, 3]
            k_data = np.loadtxt(k_file, comments="#")[:, 3]


            def calculate_rmse(y0, y1):
                if y0.shape != y1.shape:
                    raise ValueError("The files have different shapes!")
                error = y0 - y1
                mse = np.mean(error ** 2)
                rmse = np.sqrt(mse)
                return rmse

            # Calculates RMSE of k output relative to patient output
            rmse_vector = calculate_rmse(patient_data, k_data) 

            # Average RMSE
            rmse_total = np.mean(rmse_vector) 

            # Prints
            print(f"### AVERAGE RMSE (patient - k = {knumber}): {rmse_total:.6f}")

            # Stores it to plot
            rmse_means.append(rmse_total)

        # Iterations: 0 to knumber_max
        iterations = np.arange(0, knumber_max + 1)

        # Plots
        fig = plt.figure(figsize=(8, 5))
        plt.plot(iterations, rmse_means, marker='o', linestyle='-', color='tab:red')
        plt.xlabel('Iterations')
        plt.ylabel('Average RMSE (mid-point)')
        plt.title(f'Average RMSE vs Iterations - {vessel}')
        plt.grid(True)
        plt.tight_layout()

        # Creates folder for saving plots
        plots_dir = os.path.join(data_dir, f"iteration_plots_beta={beta}")
        os.makedirs(plots_dir, exist_ok=True)

        # Saves plots in .png .svg and .pdf formats
        plot_path = os.path.join(plots_dir, f"RMSE_plot")

        plt.savefig(f"{plot_path}.png", dpi=300)
        plt.savefig(f"{plot_path}.svg")
        with open(f"{plot_path}.pkl", "wb") as f:
            pickle.dump(fig, f)

        plt.close(fig)

        print(f"Plots saved: {plot_path}.png, {plot_path}.svg, {plot_path}.pkl")

    def plot_iter(self, vessel, beta, delta_dict, data_dir: str, knumber_max: int):
        """Plota os parâmetros com delta ≠ 0 e suas diferenças relativas em relação ao paciente."""

        plt.close('all')

        file_template = 'Pdk_{}.last'
        plots_dir = os.path.join(data_dir, f"iteration_plots_beta={beta}")
        os.makedirs(plots_dir, exist_ok=True)

        patient_parameters = "Pm"
        patient_yaml = "problema_inverso - Paciente.yaml"
        self.Pdk(vessel, delta_dict, patient_parameters, patient_yaml)

        file_name = file_template.format(vessel)
        folders = ['Pd0'] + [f'optimized_parameters_Pd{i}' for i in range(1, knumber_max + 1)]

        all_parameters = ["h0", "L", "R0", "E", "Rp", "Rd"]
        param_labels = {
            "h0": "h0 - Wall thickness [m]",
            "L": "L - Length [m]",
            "R0": "R0 - Lumen radius [m]",
            "E": "E - Elastic modulus [Pa]",
            "Rp": "Rp - Proximal radius [m]",
            "Rd": "Rd - Distal radius [m]"
        }

        valid_params = [p for p in all_parameters if delta_dict.get(p, 0) != 0]
        param_series = {p: [] for p in valid_params}

        for folder in folders:
            file_path = os.path.join(self.openBF_dir, folder, file_name)
            if not os.path.isfile(file_path):
                print(f"Error: File not found at {file_path}")
                return

            dados = np.loadtxt(file_path).flatten()
            for i, p in enumerate(valid_params):
                param_series[p].append(dados[i])

        patient_path = os.path.join(self.openBF_dir, patient_parameters, file_name)
        if not os.path.isfile(patient_path):
            print(f"Error: Patient file not found at {patient_path}")
            return

        patient_data = np.loadtxt(patient_path).flatten()
        patient_vals = {p: patient_data[i] for i, p in enumerate(valid_params)}

        iterations = np.arange(len(folders))

        # Separar E dos outros
        params_main = [p for p in valid_params if p != "E"]
        has_E = "E" in valid_params

        # Plot absoluto (sem E)
        if params_main:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for p in params_main:
                y = param_series[p]
                label = param_labels.get(p, p)
                line, = ax1.plot(iterations, y, 'o-', label=label)
                ax1.axhline(patient_vals[p], linestyle='--', linewidth=2,
                            color=line.get_color(), label=f'Patient {p}')
            ax1.set(title=f'Parameters vs Iterations - {vessel}', xlabel='Iterations', ylabel='Parameter Values')
            ax1.grid(True)
            ax1.legend()
            plot_path = os.path.join(plots_dir, f"{vessel}_plot_all_params")
            fig1.savefig(f"{plot_path}.png", dpi=300)
            fig1.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig1, f)
            plt.close(fig1)
            print(f"Saved: {plot_path}.png, .svg, .pkl")

        # Plot exclusivo para E
        if has_E:
            figE, axE = plt.subplots(figsize=(10, 6))
            axE.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            axE.ticklabel_format(style='plain', axis='y')
            yE = param_series["E"]
            axE.plot(iterations, yE, 'o-', color='tab:red', label=param_labels["E"])
            axE.axhline(patient_vals["E"], linestyle='--', linewidth=2, color='tab:red', label='Patient E')
            axE.set(title=f'Elastic Modulus vs Iterations - {vessel}', xlabel='Iterations',
                    ylabel='Elastic modulus [Pa]')
            axE.grid(True)
            axE.legend()
            plot_path_E = os.path.join(plots_dir, f"{vessel}_plot_E_only")
            figE.savefig(f"{plot_path_E}.png", dpi=300)
            figE.savefig(f"{plot_path_E}.svg")
            with open(f"{plot_path_E}.pkl", "wb") as f:
                pickle.dump(figE, f)
            plt.close(figE)
            print(f"Saved: {plot_path_E}.png, .svg, .pkl")

        # Plot diferenças relativas (todos juntos, inclusive E)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for p in valid_params:
            vals = np.array(param_series[p])
            ref = patient_vals[p]
            diff = (vals - ref) / ref
            ax2.plot(iterations, diff, marker='o', label=param_labels.get(p, p))
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set(title=f'Relative Difference of Parameters - {vessel}', xlabel='Iterations', ylabel='Relative Difference')
        ax2.grid(True)
        ax2.legend()
        rel_diff_path = os.path.join(plots_dir, f"{vessel}_relative_diff_plot")
        fig2.savefig(f"{rel_diff_path}.png", dpi=300)
        fig2.savefig(f"{rel_diff_path}.svg")
        with open(f"{rel_diff_path}.pkl", "wb") as f:
            pickle.dump(fig2, f)
        plt.close(fig2)
        print(f"Saved: {rel_diff_path}.png, .svg, .pkl")


    def file_openBF(self, yaml_file, output_folder_name):
        """Runs openBF in Julia for the specified YAML file;
            stacks the output values;
            and plots the graphs (Pressure/Area/Flow/Velocity vs. Time)."""

        # Where the yaml_file output files will be
        file_dir = os.path.join(openBF_dir, output_folder_name)
        os.makedirs(file_dir, exist_ok=True)

        # Runs openBF to yaml_file
        self.openBF(yaml_file, file_dir)

        # Stacking order of vessels and variables
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["P", "u"]

        # Stack openBF outputs for each vessel individually
        # Plots the simulation output graphs and saves them
        for vessel in vessels:
            self.stack_last_files(vessel, variables, file_dir)
            #self.plot_openBF(vessel, file_dir)


    def updated_openBF(self, knumber, vase, parameter, add_value):
        """ Updates the value of a parameter within a specific vessel;
        runs openBF in Julia for the updated YAML;
        stacks the output values, calculates the partial derivatives with respect to the modified parameter;
        and plots the graphs (Pressure/Area/Flow/Velocity vs. Time)."""

        # Updates the YAML file to the specified parameter
        self.update_yaml(knumber, vase, parameter, add_value)

        # Where the k_file output files are
        base_dir = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}")
        os.makedirs(base_dir, exist_ok=True)
        # Where the updated_file output files will be
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_{vase}_{parameter}")
        os.makedirs(updated_dir, exist_ok=True)

        # Runs openBF to updated_file
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}.yaml")
        self.openBF(updated_file,updated_dir)

        # Stacking order of vessels and variables
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["P", "u"]

        # Stack openBF outputs for each vessel individually
        for vessel in vessels:
            self.stack_last_files(vessel, variables, updated_dir)

        # Path to the partial derivatives directory
        del_dir = os.path.join(openBF_dir, f"partial_deriv_{parameter}")

        # Calculates and creates the partial derivatives files
        self.partial_deriv_files(vase, base_dir, updated_dir, del_dir, parameter)

        # Plots the simulation output graphs and saves them
        #self.plot_openBF(vessel, updated_dir)

    def iteration(self, knumber, vase, alpha, beta, add_h0, add_L, add_R0, add_Rp, add_Rd, add_E):
        """Creates the Jacobian pseudoinverse matrix considering the increments specified for each parameter,
        multiplies it to the y_tilde matrix and generates the optimized parameters."""
        add_values = {"h0": add_h0, "L": add_L, "R0": add_R0, "Rp": add_Rp, "Rd": add_Rd, "E": add_E}

        # Filters parameters with delta != 0
        valid_parameters = [param for param in add_values if add_values[param] != 0]
        print (f"The valid parameters are: {valid_parameters}.")

        if knumber == 0:
            k_yaml_file = os.path.join(openBF_dir, self.k0_file)

            # Checks if file exists
            if not os.path.exists(k_yaml_file):
                print(f"Error: File {k_yaml_file} not found.")
                return

            # Runs openBF to 0-iteration YAML file
            self.file_openBF(k_yaml_file, f"y{knumber} - openBF output iteration {knumber}")

        for parameter in valid_parameters:
            self.updated_openBF(knumber, vase, parameter, add_values[parameter])

        # Path to the Jacobian matrices directory
        output_path = os.path.join(openBF_dir, f"jacobians")
        self.stack_partial_derivatives(vase, add_values, output_path)

        # Creates the Pd0 matrix (parameters of the k-iteration yaml)
        if knumber == 0:
            yaml_file = self.k0_file
            param_directory = "Pd0"

            self.Pdk(vase, add_values, param_directory, yaml_file)

        # Creates the pseudoinverse matrix
        self.M_matrix(beta, vase, knumber, valid_parameters)

        # Creates the y_tilde matrix
        self.Z_matrix(vase, knumber)

        # Creates the optimized parameters matrix
        self.optimized_parameters(vase, alpha, beta, knumber)

        # Updates YAML with optimized parameters
        if knumber == 0:
            base_yaml_path = os.path.join(openBF_dir, self.k0_file)
        else:
            base_yaml_path = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")
        opt_param_files_dir = os.path.join(openBF_dir, f"optimized_parameters_Pd{knumber+1}")
        opt_output_yaml_path = os.path.join(openBF_dir, f"problema_inverso - k={knumber+1}.yaml")

        # Checks if file exists
        if not os.path.exists(base_yaml_path):
            print(f"Error: File {base_yaml_path} not found.")
            return

        self.update_yaml_with_optimized_parameters(vase, add_values, base_yaml_path, opt_param_files_dir, opt_output_yaml_path)

        # Runs openBF to the new/optimized yaml file
        self.file_openBF(opt_output_yaml_path, f"y{knumber+1} - openBF output iteration {knumber+1}")


    def search_opt(self, vase, alpha, beta, add_h0, add_L, add_R0, add_Rp, add_Rd, add_E, knumber_max):

        add_values = {"h0": add_h0, "L": add_L, "R0": add_R0, "Rp": add_Rp, "Rd": add_Rd, "E": add_E}

        # Starts chronometer
        start = time.time()

        # Runs iteration for k from 0 to knumber_max
        for knumber in range(0, knumber_max + 1):
            self.iteration(knumber, vase, alpha, beta, add_h0, add_L, add_R0, add_Rp, add_Rd, add_E)

        # Plots RMSE for k from 0 to knumber_max
        self.plot_RMSE(vase, beta, openBF_dir, knumber_max)

        # Plots the parameters for each iteration
        self.plot_iter(vase, beta, add_values, openBF_dir, knumber_max)

        # Ends chronometer and prints time
        end = time.time()
        minutes = (end - start)/60
        print(f"Elapsed time: {minutes:.3f} minutes.")




# Application
if __name__ == "__main__":

    patient_file = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_vase1/problema_inverso - Paciente.yaml"
    k0_file = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_vase1/problema_inverso - k=0 - fixed_vessels_2and3.yaml"
    openBF_dir = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_vase1"

    updater = OPENBF_Jacobian(patient_file, k0_file, openBF_dir)

    # Runs openBF to patient file
    #updater.file_openBF(patient_file, "ym - openBF output paciente")

    # Searches optimized parameters
    # search_opt(self, vase, alpha, add_h0, add_L, add_R0, add_Rp, add_Rd, add_E, knumber_max)
    
    exponents = np.arange(2, 6)  # 6 is exclusive, so it goes up to 5 (100.000)
    beta_values = 10.0 ** exponents
    alpha = 0.3
    for beta in beta_values:
        updater.search_opt("vase1", alpha, beta, 0.00001, 0.001, 0.0001, 0, 0, 0, 20)
