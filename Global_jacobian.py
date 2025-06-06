import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time

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
    def __init__(self, patient_file, k0_file, openBF_dir):
        self.patient_file = patient_file
        self.k0_file = k0_file # Is not used, but needs to exist
        self.openBF_dir = openBF_dir


    def update_yaml(self, knumber, vase, parameter, add_value):

        self.add_value = add_value
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}_{vase}.yaml")

        # Loads YAML from k_file
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

    def stack_vessels_files(self, data_dir):
        """Stacks the openBF output of all vessels in the order of the vessels from top to bottom
        and saves it to a .last file"""
        vessels = ["vase1", "vase2", "vase3"]

        data_list = []

        for vessel in vessels:
            file_path = os.path.join(data_dir, f"{vessel}_stacked.last")

            # Checks if file exists
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found.")
                return

            # Reads data from the .last file
            data = np.loadtxt(file_path)
            data_list.append(data)

        # Stacks files vertically (vase1, vase2, vase3)
        stacked_data = np.vstack(data_list)

        # Saves the stacked file
        stacked_file = os.path.join(data_dir, f"vessels_stacked.last")
        np.savetxt(stacked_file, stacked_data, fmt="%.14e")

        print(f"Saved file: {stacked_file}")

    def partial_deriv_files(self, base_dir, updated_dir, del_dir, vessel, parameter, delta_value):
        """
        Subtracts the stacked files of base_dir from the stacked files of updated_dir,
        divides by the delta parameter and saves the result in del_dir.
        """

        print(f"Parameter '{parameter}' difference: {delta_value} for {vessel}")

        if delta_value == 0:
            print(f"Warning: Parameter difference for {parameter} is zero. Avoiding division by zero.")
            return

        # Cria diretório de saída se não existir
        os.makedirs(del_dir, exist_ok=True)

        base_file_path = os.path.join(base_dir, f"vessels_stacked.last")
        updated_file_path = os.path.join(updated_dir, f"vessels_stacked.last")
        del_file_path = os.path.join(del_dir, f"{vessel}_del_{parameter}_delta={delta_value}.last")

        # Verifica existência
        if not os.path.exists(base_file_path):
            print(f"Error: Base file not found: {base_file_path}")
            return

        if not os.path.exists(updated_file_path):
            print(f"Error: Updated file not found: {updated_file_path}")
            return

        # Carrega dados
        base_data = np.loadtxt(base_file_path)
        updated_data = np.loadtxt(updated_file_path)

        # Verifica dimensões
        if base_data.shape != updated_data.shape:
            print(
                f"Error: Incompatible dimensions for {parameter} ({vessel}): {base_data.shape} vs {updated_data.shape}.")
            return

        # Calcula derivada parcial
        del_data = (updated_data - base_data) / delta_value

        # Salva resultado
        np.savetxt(del_file_path, del_data, fmt="%.14e")
        print(f"Partial derivatives file saved: {del_file_path}")

    def plot_openBF(self, data_dir):
        """ Plots the openBF output graphs (Pressure/Area/Flow/Velocity vs. Time)
        and saves them in data_dir.
        """
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["P", "u"]
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

            plt.close(fig)

            print(f"Plots saved: {plot_path}.png, {plot_path}.svg, {plot_path}.pkl")


    def stack_global_jacobian(self, delta_dict):
        parameters = ["h0", "L", "R0"]
        vessels = ["vase1", "vase2", "vase3"]

        global_columns = []  # Lista final com as 9 coluna

        for vessel in vessels:
            for param in parameters:
                if delta_dict[param] == 0:
                    print(f"Skipping parameter {param} due to zero delta.")
                    continue

                delta = delta_dict[param]
                file_path = os.path.join(openBF_dir, f"partial_deriv_{param}",
                                         f"{vessel}_del_{param}_delta={delta}.last")

                if os.path.exists(file_path):
                    matrix = np.loadtxt(file_path)

                    if matrix.ndim == 1:
                        print(f"Error: Partial derivative matrix {file_path} has only 1 dimension.")
                        return

                    col = matrix[:, 3].reshape(-1, 1)  # Pega a quarta coluna
                    global_columns.append(col)  # Agora cada coluna → derivada de 1 vaso + 1 param
                else:
                    raise FileNotFoundError(f"Missing file: {file_path}")

        # Empilha todas as colunas horizontalmente → shape: (600,9)
        J_global = np.hstack(global_columns)

        output_file = os.path.join(self.openBF_dir, "jacobian_global.last")
        np.savetxt(output_file, J_global, fmt="%.14e")
        print(f"Global Jacobian saved in: {output_file}, shape: {J_global.shape}")



    def pseudoinverse_matrix(self):
        """ Creates the pseudoinverse_matrix using the Jacobian matrix and saves it in the folder: "jacobians_pseudoinverse"."""

        file_path = os.path.join(openBF_dir, f"jacobian_global.last")

        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path)  # Loads the Jacobian matrix
            JkT_Jk = Jk.T @ Jk  # Jacobian transpose times the Jacobian

            # Checks if it is invertible
            # Calculates the conditional number
            cond_matrix = np.linalg.cond(JkT_Jk)
            cond_Jk = np.linalg.cond(Jk)

            # Sets a threshold to consider "non-invertible"
            threshold = 1e6

            print(f"Conditional number of the Jk matrix: {cond_Jk:.2e}")

            if cond_Jk < threshold:
                print("The Jk matrix is invertible.")
            else:
                print("Error: The Jk matrix is quasi-singular or non-invertible.")

            print(f"Conditional number of the inverse matrix: {cond_matrix:.2e}")

            if cond_matrix < threshold:
                print("The JkT@Jk matrix is invertible.")
            else:
                print("Error: The JkT@Jk matrix is quasi-singular or non-invertible.")

            # Calculates the pseudoinverse matrix
            JkT_Jk_inv = np.linalg.inv(JkT_Jk)
            pseudoinv = JkT_Jk_inv @ Jk.T

            output_file = os.path.join(openBF_dir, f"pseudoinv_matrix.last")
            np.savetxt(output_file, pseudoinv, fmt="%.14e")
            print(f"Pseudoinverse matrix saved in: {output_file}")

        else:
            print(f"Error: File not found - {file_path}.")
            return


    def y_tilde(self, knumber):
        ''' Constructs the y-tilde matrix: compute the difference between the openBF output from the
        iteration k .yaml file and the openBF output from the patient's .yaml file'''

        # Loads the data from the patient openBF output - ym
        patient_output = os.path.join(openBF_dir, "ym - openBF output paciente", "vessels_stacked.last")

        if os.path.exists(patient_output):
            data = np.loadtxt(patient_output)
            patient_data = data[:, 3].reshape(-1, 1)  # Just takes the 3rd knot (4th column)
        else:
            print(f"Error: Patient output file not found - {patient_output}.")
            return

        # Loads the simulation output corresponding to the guess
        yk_output = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}",
                                 "vessels_stacked.last")

        if os.path.exists(yk_output):
            data = np.loadtxt(yk_output)
            yk_data = data[:, 3].reshape(-1, 1)  # Just takes the 3rd knot (4th column)
        else:
            print(f"Error: Iteration output file not found - {yk_output}.")
            return

        y_tilde = patient_data - yk_data

        # Saves the y_tilde matrix in a file
        y_tilde_file = os.path.join(openBF_dir, f"y_tilde.last")
        np.savetxt(y_tilde_file, y_tilde, fmt="%.14e")
        print(f"y_tilde matrix saved: {y_tilde_file}")


    def Pdk(self, param_directory, yaml_file):
        """
        Loads the parameters of vessels from a YAML file and saves them stacked vertically in a single file.
        """

        vessels = ["vase1", "vase2", "vase3"]
        parameters = ["h0", "L", "R0"]
        Pdk = []

        # Path to the YAML file
        k_file = os.path.join(openBF_dir, yaml_file)
        with open(k_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the YAML.")
            return

        for vessel in vessels:
            vessel_params = []

            found = False  # Flag to know if the vessel was found

            for item in yaml_data["network"]:
                if item.get("label") == vessel:
                    for parameter in parameters:
                        if parameter in item:
                            value = item[parameter]
                            vessel_params.append(value)
                            found = True
                        else:
                            print(f"Error: Parameter '{parameter}' not found in vessel '{vessel}'.")
                            return

            if not found:
                print(f"Error: Vessel '{vessel}' not found in YAML.")
                return

            # Reshape as column vector and append to the list
            Pdk.append(np.array(vessel_params).reshape(-1, 1))

        # Stack all vessel parameter vectors vertically
        Pdk_stacked = np.vstack(Pdk)

        # Where the parameters file will be
        file_dir = os.path.join(openBF_dir, param_directory)
        os.makedirs(file_dir, exist_ok=True)

        # Save the stacked parameter vector in a single file
        Pdk_file = os.path.join(file_dir, "Pdk_stacked.last")
        np.savetxt(Pdk_file, Pdk_stacked, fmt="%.14e")
        print(f"Stacked parameter vector saved: {Pdk_file}")


    def optimized_parameters_global(self, knumber, alpha=0.3):
        vessels = ["vase1", "vase2", "vase3"]
        parameters = ["h0", "L", "R0"]

        # Load pseudoinverse
        pseudo_inv_path = os.path.join(openBF_dir, "pseudoinv_matrix.last")
        if not os.path.exists(pseudo_inv_path):
            print(f"Error: Pseudoinverse global file not found - {pseudo_inv_path}")
            return
        pseudo_inv = np.loadtxt(pseudo_inv_path)

        # Load y_tilde global
        y_tilde_path = os.path.join(openBF_dir, "y_tilde.last")
        if not os.path.exists(y_tilde_path):
            print(f"Error: y_tilde global file not found - {y_tilde_path}")
            return
        y_tilde = np.loadtxt(y_tilde_path).reshape(-1, 1)

        delta_params = pseudo_inv @ y_tilde  # (9x1)

        # Creates directory
        opt_param_dir = os.path.join(openBF_dir, f"optimized_parameters_Pd{knumber + 1}")
        os.makedirs(opt_param_dir, exist_ok=True)

        # Load current params
        if knumber == 0:
            param_path = os.path.join(openBF_dir, f"Pd{knumber}", f"Pdk_stacked.last")
        else:
            param_path = os.path.join(openBF_dir, f"optimized_parameters_Pd{knumber}", f"Pdk_stacked.last")

        if os.path.exists(param_path):
            param_data = np.loadtxt(param_path).reshape(-1, 1)
        else:
            print(f"Error: Pdk matrix file not found - {param_path}")
            return

        # Atualização com sub-relaxação
        new_param = param_data + alpha * delta_params

        opt_param_file = os.path.join(opt_param_dir, f"Pdk_stacked.last")
        np.savetxt(opt_param_file, new_param, fmt="%.14e")
        print(f"Optimized parameters saved: {opt_param_file}")

    def update_yaml_with_optimized_parameters(self, base_yaml_path, param_files_dir, output_yaml_path):
        """
        Updates the input YAML using the optimized parameters from a single stacked file (Pdk_stacked.last).
        """

        vessels = ["vase1", "vase2", "vase3"]
        parameters = ["h0", "L", "R0"]
        num_params_per_vessel = len(parameters)

        # Loads the YAML file
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: 'network' key not found in YAML.")
            return

        # Loads the stacked optimized parameters
        param_file = os.path.join(param_files_dir, "Pdk_stacked.last")
        if not os.path.exists(param_file):
            print(f"Error: Parameter file not found - {param_file}.")
            return

        stacked_params = np.loadtxt(param_file).flatten()

        if len(stacked_params) != num_params_per_vessel * len(vessels):
            print(
                f"Error: Number of parameters mismatch. Expected {num_params_per_vessel * len(vessels)}, got {len(stacked_params)}.")
            return

        # Iterates through each vessel and updates its parameters
        for i, vessel in enumerate(vessels):
            start_idx = i * num_params_per_vessel
            end_idx = start_idx + num_params_per_vessel
            vessel_params = stacked_params[start_idx:end_idx]

            # Updates the YAML values
            for item in yaml_data["network"]:
                if item.get("label") == vessel:
                    for j, param in enumerate(parameters):
                        item[param] = float(vessel_params[j])
                    print(f"Updated parameters for {vessel}: {vessel_params}")
                    break
            else:
                print(f"Warning: Vessel {vessel} not found in YAML.")

        # Saves the new YAML file
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated YAML saved in: {output_yaml_path}")


    def plot_RMSE(self, data_dir, knumber_max):
        # Plots the global RMSE (vessels_stacked) vs. iteration.

        plt.close('all')
        rmse_means = []

        for knumber in range(0, knumber_max + 1):

            # Files paths
            patient_file = os.path.join(openBF_dir, "ym - openBF output paciente", "vessels_stacked.last")
            k_file = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}", "vessels_stacked.last")

            # Checks if files exist
            if not os.path.exists(patient_file):
                print(f"Error: File {patient_file} not found.")
                return
            if not os.path.exists(k_file):
                print(f"Error: File {k_file} not found.")
                return

            # Loads stacked files ignoring comments
            patient_data = np.loadtxt(patient_file, comments="#")[:, 1:]  # Ignores the 1st column (time)
            k_data = np.loadtxt(k_file, comments="#")[:, 1:]

            # Calculates global RMSE
            def calculate_rmse(y0, y1):
                if y0.shape != y1.shape:
                    raise ValueError(f"Shape mismatch: {y0.shape} vs {y1.shape}")
                error = y0 - y1
                mse = np.mean(error ** 2, axis=0)  # RMSE por coluna
                rmse = np.sqrt(mse)
                return rmse

            rmse_columns = calculate_rmse(patient_data, k_data)

            # Mean RMSE across all columns
            mean_rmse = np.mean(rmse_columns)
            print(f"### GLOBAL RMSE (k = {knumber} - patient): {mean_rmse:.6f}")

            # Stores it
            rmse_means.append(mean_rmse)

        # Iterations: 0 to knumber_max
        iterations = np.arange(0, knumber_max + 1)

        # Plots
        fig = plt.figure(figsize=(8, 5))
        plt.plot(iterations, rmse_means, marker='o', linestyle='-', color='tab:red')
        plt.xlabel('Iterations')
        plt.ylabel('Global RMSE')
        plt.title('Global RMSE vs Iterations')
        plt.grid(True)
        plt.tight_layout()

        # Creates folder for saving plots
        plots_dir = os.path.join(data_dir, "iteration plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Saves plots in .png .svg and .pkl formats
        plot_path = os.path.join(plots_dir, f"RMSE_plot")

        plt.savefig(f"{plot_path}.png", dpi=300)
        plt.savefig(f"{plot_path}.svg")
        with open(f"{plot_path}.pkl", "wb") as f:
            pickle.dump(fig, f)

        plt.close(fig)

        print(f"Plots saved: {plot_path}.png, {plot_path}.svg, {plot_path}.pkl")

    def plot_iter(self, data_dir, knumber_max):
        """Plots parameter values and relative differences versus iterations, compared to patient parameters."""

        plt.close('all')

        vessels = ["vase1", "vase2", "vase3"]
        titles = ["Vessel 1", "Vessel 2", "Vessel 3"]
        parameters = ["h0", "L", "R0"]
        num_params_per_vessel = len(parameters)

        patient_parameters = "Pm"
        patient_yaml = "problema_inverso - Paciente.yaml"
        self.Pdk(patient_parameters, patient_yaml)

        # Load patient parameters (stacked)
        patient_stacked_file = os.path.join(openBF_dir, patient_parameters, "Pdk_stacked.last")
        if not os.path.isfile(patient_stacked_file):
            print(f"Error: Patient stacked parameters file not found at {patient_stacked_file}")
            return
        patient_data = np.loadtxt(patient_stacked_file).flatten()

        plots_dir = os.path.join(data_dir, "iteration plots")
        os.makedirs(plots_dir, exist_ok=True)

        for i, (vessel, title) in enumerate(zip(vessels, titles)):
            plt.close('all')

            # Initialize parameter value lists
            h0_list, L_list, R0_list = [], [], []

            # Folder list
            folders = ['Pd0'] + [f'optimized_parameters_Pd{j}' for j in range(1, knumber_max + 1)]

            for folder in folders:
                stacked_file = os.path.join(openBF_dir, folder, "Pdk_stacked.last")

                if os.path.isfile(stacked_file):
                    stacked_data = np.loadtxt(stacked_file).flatten()
                    start_idx = i * num_params_per_vessel
                    end_idx = start_idx + num_params_per_vessel
                    vessel_params = stacked_data[start_idx:end_idx]

                    h0_list.append(vessel_params[0])
                    L_list.append(vessel_params[1])
                    R0_list.append(vessel_params[2])
                else:
                    print(f"Error: File not found at {stacked_file}")
                    return

            # Get patient values
            start_idx = i * num_params_per_vessel
            end_idx = start_idx + num_params_per_vessel
            h0_patient, L_patient, R0_patient = patient_data[start_idx:end_idx]

            # Convert to numpy for computation
            h0_array = np.array(h0_list)
            L_array = np.array(L_list)
            R0_array = np.array(R0_list)

            # Compute relative differences
            diff_h0 = (h0_array - h0_patient) / h0_patient
            diff_L = (L_array - L_patient) / L_patient
            diff_R0 = (R0_array - R0_patient) / R0_patient

            iterations = np.arange(len(h0_list))

            # Plot: Absolute parameter values
            fig1, ax1 = plt.subplots(figsize=(10, 6))

            ax1.plot(iterations, h0_list, 'o-', label='h0 - Wall thickness')
            ax1.plot(iterations, L_list, 's-', label='L - Length')
            ax1.plot(iterations, R0_list, '^-', label='R0 - Lumen radius')

            ax1.axhline(h0_patient, color='tab:blue', linestyle='--', linewidth=2, label='Patient h0')
            ax1.axhline(L_patient, color='tab:orange', linestyle='--', linewidth=2, label='Patient L')
            ax1.axhline(R0_patient, color='tab:green', linestyle='--', linewidth=2, label='Patient R0')

            ax1.set(title=f'Parameters vs Iterations - {title}', xlabel='Iterations', ylabel='Parameter Values')
            ax1.grid(True)
            ax1.legend()

            plot_path1 = os.path.join(plots_dir, f"{vessel}_plot")
            fig1.savefig(f"{plot_path1}.png", dpi=300)
            fig1.savefig(f"{plot_path1}.svg")
            with open(f"{plot_path1}.pkl", "wb") as f:
                pickle.dump(fig1, f)
            plt.close(fig1)

            print(f"Saved: {plot_path1}.png, .svg, .pkl")

            # Plot: Relative differences
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            ax2.plot(iterations, diff_h0, 'o-', label='h0 - Wall thickness')
            ax2.plot(iterations, diff_L, 's-', label='L - Length')
            ax2.plot(iterations, diff_R0, '^-', label='R0 - Lumen radius')

            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

            ax2.set(title=f'Relative Parameter Difference - {title}',
                    xlabel='Iterations', ylabel='Relative Difference')
            ax2.grid(True)
            ax2.legend()

            plot_path2 = os.path.join(plots_dir, f"{vessel}_relative_diff_plot")
            fig2.savefig(f"{plot_path2}.png", dpi=300)
            fig2.savefig(f"{plot_path2}.svg")
            with open(f"{plot_path2}.pkl", "wb") as f:
                pickle.dump(fig2, f)
            plt.close(fig2)

            print(f"Saved: {plot_path2}.png, .svg, .pkl")

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
        parameters = ["h0", "L", "R0"]

        # Stack openBF outputs for each vessel individually
        for vessel in vessels:
            self.stack_last_files(vessel, variables, file_dir)

        # Stack 3 vessels files to make the jacobian columns
        self.stack_vessels_files(file_dir)

        # Plots the simulation output graphs and saves them
        #self.plot_openBF(file_dir)


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
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_{parameter}_{vase}")
        os.makedirs(updated_dir, exist_ok=True)

        # Runs openBF to updated_file
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}_{vase}.yaml")
        self.openBF(updated_file,updated_dir)

        # Stacking order of vessels and variables
        vessels = ["vase1", "vase2", "vase3"]
        variables = ["P", "u"]
        parameters = ["h0", "L", "R0"]

        # Stack openBF outputs for each vessel individually
        for vessel in vessels:
            self.stack_last_files(vessel, variables, updated_dir)

        # Stack 3 vessels files to make the jacobian columns
        self.stack_vessels_files(updated_dir)

        # Plots the simulation output graphs and saves them
        #self.plot_openBF(updated_dir)

    def iteration(self, knumber, add_h0, add_L, add_R0):
        """Creates the Jacobian pseudoinverse matrix considering the increments specified for each parameter,
        multiplies it to the y_tilde matrix and generates the optimized parameters."""
        add_values = {"h0": add_h0, "L": add_L, "R0": add_R0}
        vessels = ["vase1", "vase2", "vase3"]

        if knumber == 0:
            k_yaml_file = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")

            # Checks if file exists
            if not os.path.exists(k_yaml_file):
                print(f"Error: File {k_yaml_file} not found.")
                return

            # Runs openBF to 0-iteration YAML file
            self.file_openBF(k_yaml_file, f"y{knumber} - openBF output iteration {knumber}")

        for parameter in add_values:
            for vessel in vessels:
                self.updated_openBF(knumber, vessel, parameter, add_values[parameter])

                # Calculates and creates the partial derivatives files
                base_dir = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}")
                updated_dir = os.path.join(self.openBF_dir, f"openBF_updated_{parameter}_{vessel}")
                del_dir = os.path.join(self.openBF_dir, f"partial_deriv_{parameter}")
                self.partial_deriv_files(base_dir, updated_dir, del_dir, vessel, parameter, add_values[parameter])

        # Path to the Jacobian matrices directory
        self.stack_global_jacobian(add_values)

        # Creates the Pd0 matrix (parameters of the k-iteration yaml)
        if knumber == 0:
            yaml_file = "problema_inverso - k=0.yaml"
            param_directory = "Pd0"

            self.Pdk(param_directory, yaml_file)

        # Pseudoinverse
        self.pseudoinverse_matrix()

        # liberar depois
        # y_tilde
        self.y_tilde(knumber)

        # Optimized parameters
        self.optimized_parameters_global(knumber)

        # Atualiza YAML
        base_yaml_path = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")
        opt_param_files_dir = os.path.join(openBF_dir, f"optimized_parameters_Pd{knumber + 1}")
        opt_output_yaml_path = os.path.join(openBF_dir, f"problema_inverso - k={knumber + 1}.yaml")
        self.update_yaml_with_optimized_parameters(base_yaml_path, opt_param_files_dir, opt_output_yaml_path)

        # Run openBF
        self.file_openBF(opt_output_yaml_path, f"y{knumber + 1} - openBF output iteration {knumber + 1}")

    def search_opt(self, add_h0, add_L, add_R0, knumber_max):

        # Starts chronometer
        start = time.time()

        # Runs iteration for k from 0 to knumber_max
        for knumber in range(0, knumber_max + 1):
            self.iteration(knumber, add_h0, add_L, add_R0)

        # Plots RMSE for k from 0 to knumber_max
        self.plot_RMSE(openBF_dir, knumber_max)

        # Plots the parameters for each iteration
        self.plot_iter(openBF_dir, knumber_max)

        # Ends chronometer and prints time
        end = time.time()
        minutes = (end - start)/60
        print(f"Elapsed time: {minutes:.3f} minutes.")




# Application
if __name__ == "__main__":

    patient_file = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_global/problema_inverso - Paciente.yaml"
    k0_file = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_global/problema_inverso - k=0.yaml"
    openBF_dir = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_global"

    updater = OPENBF_Jacobian(patient_file, k0_file, openBF_dir)

    # Runs openBF to patient file
    #updater.file_openBF(patient_file, "ym - openBF output paciente")

    # Searches optimized parameters
    #updater.search_opt(0.00001,0.0001, 0.0001, 20)
    updater.plot_iter(openBF_dir, 20)
