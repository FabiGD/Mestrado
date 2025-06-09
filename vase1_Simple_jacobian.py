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
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}.yaml")

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

            df = data[var]
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
        parameters = ["h0", "L", "R0"]

        # Filtra os parâmetros com delta != 0
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

    def pseudoinverse_matrix(self, vessel):
        """ Creates the pseudoinverse_matrix using the Jacobian matrix and saves it in the folder: "jacobians_pseudoinverse"."""

        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_{vessel}_stacked.txt")

        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
            JkT_Jk = Jk.T @ Jk # Jacobian transpose times the Jacobian

            # Checks if it is invertible
            # Calculates the conditional number
            cond_matrix = np.linalg.cond(JkT_Jk)
            cond_Jk = np.linalg.cond(Jk)

            # Sets a threshold to consider "non-invertible"
            threshold = 1e6

            print(f"Conditional number of the Jk matrix ({vessel}): {cond_Jk:.2e}")

            if cond_Jk < threshold:
                print("The Jk matrix is invertible.")
            else:
                print("Error: The Jk matrix is quasi-singular or non-invertible.")

            print(f"Conditional number of the inverse matrix ({vessel}): {cond_matrix:.2e}")

            if cond_matrix < threshold:
                print("The JkT@Jk matrix is invertible.")
            else:
                print("Error: The JkT@Jk matrix is quasi-singular or non-invertible.")



            # Calculates the pseudoinverse matrix
            JkT_Jk_inv = np.linalg.inv(JkT_Jk)
            pseudoinv = JkT_Jk_inv @ Jk.T

            output_dir = os.path.join(openBF_dir, "jacobians_pseudoinverse")
            os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it does not exist

            output_file = os.path.join(output_dir, f"pseudoinv_{vessel}.txt")
            np.savetxt(output_file, pseudoinv, fmt="%.14e")
            print(f"Pseudoinverse matrix saved in: {output_file}")

        else:
            print(f"Error: File not found - {file_path}.")
            return

    def y_til(self, vessel, knumber):


        # Loads the data from the patient openBF output - ym
        patient_output = os.path.join(openBF_dir, f"ym - openBF output paciente", f"{vessel}_stacked.last")

        if os.path.exists(patient_output):
            data = np.loadtxt(patient_output)
            patient_data = data[:, 3].reshape(-1, 1) # Pego apenas o 3° nó
        else:
            print(f"Error: Patient output file not found - {patient_output}.")
            return

        # Loads the simulation output corresponding to the guess
        yk_output = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}", f"{vessel}_stacked.last")

        if os.path.exists(yk_output):
            data = np.loadtxt(yk_output)
            yk_data = data[:, 3].reshape(-1, 1)  # Pego apenas o 3° nó
        else:
            print(f"Error: Iteration output file not found - {yk_output}.")
            return

        y_til = patient_data - yk_data

        # Where the y_til matrix files will be
        y_til_dir = os.path.join(openBF_dir, "y_til")
        os.makedirs(y_til_dir, exist_ok=True)

        # Saves the y_til matrix in a file
        y_til_file = os.path.join(y_til_dir, f"y_til_{vessel}.last")
        np.savetxt(y_til_file, y_til, fmt="%.14e")
        print(f"y_til matrix saved: {y_til_file}")

    def Pdk(self, vessel, param_directory, yaml_file):
        # Loads the parameters of a yaml file and saves it in a directory

        #Pdk = {vessel: [] for vessel}
        Pdk = []

        # Creates a vector with the parameter values corresponding to the guess
        parameters = ["h0", "L", "R0"]

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
                for parameter in parameters:
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


    def optimized_parameters(self, vessel, knumber):

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

        # Loads the data from the pseudoinverse matrix
        pseudoinv_path = os.path.join(openBF_dir, "jacobians_pseudoinverse", f"pseudoinv_{vessel}.txt")

        if os.path.exists(pseudoinv_path):
            pseudoinv_data = np.loadtxt(pseudoinv_path)
        else:
            print(f"Error: Pseudoinverse matrix file not found - {pseudoinv_path}.")
            return

        # Loads the data from the y_til matrix
        y_til_path = os.path.join(openBF_dir, "y_til", f"y_til_{vessel}.last")

        if os.path.exists(y_til_path):
            y_til_data = np.loadtxt(y_til_path).reshape(-1, 1)

        else:
            print(f"Error: y_til matrix file not found - {y_til_path}.")
            return

        # Sub-relaxation factor for the Newton step
        alpha = 0.3

        # Creates the optimized parameters (Pd(k+1)) matrix
        vector_product = pseudoinv_data @ y_til_data
        opt_param_data = param_data + alpha * vector_product

        # Saves the optimized parameters matrix in a file
        opt_param_file = os.path.join(opt_param_dir, f"Pdk_{vessel}.last")
        np.savetxt(opt_param_file, opt_param_data, fmt="%.14e")
        print(f"Optimized parameters matrix saved: {opt_param_file}")


    def update_yaml_with_optimized_parameters(self, vessel, base_yaml_path, param_files_dir, output_yaml_path):
        # Updates the input YAML using the optimized parameters saved in separate files.

        parameters = ["h0", "L", "R0"]

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

        # Ensures that new_params is a vector (not an array)
        new_params = np.atleast_1d(new_params)

        if len(new_params) != len(parameters):
            print(
                f"Error: Number of parameters mismatch for {vessel}. Expected {len(parameters)}, got {len(new_params)}.")

        # Updates the YAML values
        for item in yaml_data["network"]:
            if item.get("label") == vessel:
                for i, param in enumerate(parameters):
                    item[param] = float(new_params[i])
                print(f"Updated parameters for {vessel}: {new_params}")
                break
        else:
            print(f"Warning: Vessel {vessel} not found in YAML.")

        # Saves the new YAML file
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated YAML saved in: {output_yaml_path}")

    def plot_RMSE(self, vessel, data_dir, knumber_max=6):
        # Plots the average RMSE (3 vessels) vs. iteration

        plt.close('all')
        rmse_means = []

        for knumber in range(0, knumber_max + 1):

            rmse_totals_k = []  # List to store the total RMSE of each vessel

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

            # Loads stacked files ignoring comments
            patient_data = np.loadtxt(patient_file, comments="#")[:, 1:]  # Ignores the 1st column (time)
            k_data = np.loadtxt(k_file, comments="#")[:, 1:]

            def calculate_rmse(y0, y1):
                if y0.shape != y1.shape:
                    raise ValueError("The files have different shapes!")
                error = y0 - y1
                mse = np.mean(error ** 2, axis=0)  # per column
                rmse = np.sqrt(mse)
                return rmse

            ## Calculates column-wise RMSE of k output relative to patient output
            rmse_columns_k = calculate_rmse(patient_data, k_data)

            # Average RMSE per vessel
            rmse_total_k = np.mean(rmse_columns_k)
            rmse_totals_k.append(rmse_total_k)

            # Calculation of the mean RMSEs of the 3 vessels for k
            mean_rmse_vessels_k = np.mean(rmse_totals_k)
            print(f"### AVERAGE RMSE (k = {knumber} - patient) (3 vessels): {mean_rmse_vessels_k:.6f}")

            # Stores it
            rmse_means.append(mean_rmse_vessels_k)

        # Iterations: 0 to knumber_max
        iterations = np.arange(0, knumber_max + 1)

        # Plots
        fig = plt.figure(figsize=(8, 5))
        plt.plot(iterations, rmse_means, marker='o', linestyle='-', color='tab:red')
        plt.xlabel('Iterations')
        plt.ylabel('Average RMSE (3 vessels)')
        plt.title('Average RMSE vs Iterations')
        plt.grid(True)
        plt.tight_layout()

        # Creates folder for saving plots
        plots_dir = os.path.join(data_dir, "iteration plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Saves plots in .png .svg and .pdf formats
        plot_path = os.path.join(plots_dir, f"RMSE_plot")

        plt.savefig(f"{plot_path}.png", dpi=300)
        plt.savefig(f"{plot_path}.svg")
        with open(f"{plot_path}.pkl", "wb") as f:
            pickle.dump(fig, f)

        plt.close(fig)

        print(f"Plots saved: {plot_path}.png, {plot_path}.svg, {plot_path}.pkl")

    def plot_iter(self, vessel, data_dir: str, knumber_max: int):
        """Plots parameter values vs. iterations and their relative differences compared to patient data."""

        plt.close('all')

        file_template = 'Pdk_{}.last'
        plots_dir = os.path.join(data_dir, "iteration plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Load patient parameters
        patient_parameters = "Pm"
        patient_yaml = "problema_inverso - Paciente.yaml"
        self.Pdk(vessel, patient_parameters, patient_yaml)

        plt.close('all')
        file_name = file_template.format(vessel)

        # Build list of folder names in order
        folders = ['Pd0'] + [f'optimized_parameters_Pd{i}' for i in range(1, knumber_max + 1)]

        h0_list, L_list, R0_list = [], [], []

        for folder in folders:
            file_path = os.path.join(openBF_dir, folder, file_name)

            if not os.path.isfile(file_path):
                print(f"Error: File not found at {file_path}")
                return

            dados = np.loadtxt(file_path)
            h0_list.append(dados[0])
            L_list.append(dados[1])
            R0_list.append(dados[2])

        # Load patient data
        patient_path = os.path.join(openBF_dir, patient_parameters, file_name)
        if not os.path.isfile(patient_path):
            print(f"Error: Patient file not found at {patient_path}")
            return

        patient_data = np.loadtxt(patient_path)
        h0_patient, L_patient, R0_patient = patient_data[:3]

        # Relative differences
        h0_list_np = np.array(h0_list)
        L_list_np = np.array(L_list)
        R0_list_np = np.array(R0_list)

        diff_h0 = (h0_list_np - h0_patient) / h0_patient
        diff_L = (L_list_np - L_patient) / L_patient
        diff_R0 = (R0_list_np - R0_patient) / R0_patient

        iterations = np.arange(len(h0_list))

        # Plot: Absolute values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, h0_list, 'o-', label='h0 - Wall thickness')
        ax.plot(iterations, L_list, 's-', label='L - Length')
        ax.plot(iterations, R0_list, '^-', label='R0 - Lumen radius')

        ax.axhline(h0_patient, color='tab:blue', linestyle='--', linewidth=2, label='Patient h0')
        ax.axhline(L_patient, color='tab:orange', linestyle='--', linewidth=2, label='Patient L')
        ax.axhline(R0_patient, color='tab:green', linestyle='--', linewidth=2, label='Patient R0')

        ax.set(title=f'Parameters vs Iterations - {vessel}', xlabel='Iterations', ylabel='Parameters')
        ax.grid(True)
        ax.legend()

        plot_path = os.path.join(plots_dir, f"{vessel}_plot")
        fig.savefig(f"{plot_path}.png", dpi=300)
        fig.savefig(f"{plot_path}.svg")
        with open(f"{plot_path}.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.close(fig)

        print(f"Saved: {plot_path}.png, .svg, .pkl")

        # Plot: Relative differences
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(iterations, diff_h0, 'o-', label='h0 - Wall thickness')
        ax2.plot(iterations, diff_L, 's-', label='L - Length')
        ax2.plot(iterations, diff_R0, '^-', label='R0 - Lumen radius')

        ax2.set(title=f'Relative Difference of Parameters - {vessel}',
                xlabel='Iterations', ylabel='Relative Difference')
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

    def iteration(self, knumber, vase, add_h0, add_L, add_R0):
        """Creates the Jacobian pseudoinverse matrix considering the increments specified for each parameter,
        multiplies it to the y_til matrix and generates the optimized parameters."""
        add_values = {"h0": add_h0, "L": add_L, "R0": add_R0}

        if knumber == 0:
            k_yaml_file = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")

            # Checks if file exists
            if not os.path.exists(k_yaml_file):
                print(f"Error: File {k_yaml_file} not found.")
                return

            # Runs openBF to 0-iteration YAML file
            self.file_openBF(k_yaml_file, f"y{knumber} - openBF output iteration {knumber}")

        for parameter in add_values:
            self.updated_openBF(knumber, vase, parameter, add_values[parameter])

        # Path to the Jacobian matrices directory
        output_path = os.path.join(openBF_dir, f"jacobians")
        self.stack_partial_derivatives(vase, add_values, output_path)

        # Creates the Pd0 matrix (parameters of the k-iteration yaml)
        if knumber == 0:
            yaml_file = "problema_inverso - k=0.yaml"
            param_directory = "Pd0"

            self.Pdk(vase, param_directory, yaml_file)

        # Creates the pseudoinverse matrix
        self.pseudoinverse_matrix(vase)

        # Creates the y_til matrix
        self.y_til(vase, knumber)

        # Creates the optimized parameters matrix
        self.optimized_parameters(vase, knumber)

        # Updates YAML with optimized parameters
        base_yaml_path = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")
        opt_param_files_dir = os.path.join(openBF_dir, f"optimized_parameters_Pd{knumber+1}")
        opt_output_yaml_path = os.path.join(openBF_dir, f"problema_inverso - k={knumber+1}.yaml")

        # Checks if file exists
        if not os.path.exists(base_yaml_path):
            print(f"Error: File {base_yaml_path} not found.")
            return

        self.update_yaml_with_optimized_parameters(vase, base_yaml_path, opt_param_files_dir, opt_output_yaml_path)

        # Runs openBF to the new/optimized yaml file
        self.file_openBF(opt_output_yaml_path, f"y{knumber+1} - openBF output iteration {knumber+1}")


    def search_opt(self, vase, add_h0, add_L, add_R0, knumber_max):

        # Starts chronometer
        start = time.time()

        # Runs iteration for k from 0 to knumber_max
        for knumber in range(0, knumber_max + 1):
            self.iteration(knumber, vase, add_h0, add_L, add_R0)

        # Plots RMSE for k from 0 to knumber_max
        self.plot_RMSE(vase, openBF_dir, knumber_max)

        # Plots the parameters for each iteration
        self.plot_iter(vase, openBF_dir, knumber_max)

        # Ends chronometer and prints time
        end = time.time()
        minutes = (end - start)/60
        print(f"Elapsed time: {minutes:.3f} minutes.")




# Application
if __name__ == "__main__":

    patient_file = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_vase1/problema_inverso - Paciente.yaml"
    k0_file = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_vase1/problema_inverso - k=0.yaml"
    openBF_dir = "C:/Users/Reinaldo/Documents/problema_inverso_results_openbf_vase1"

    updater = OPENBF_Jacobian(patient_file, k0_file, openBF_dir)

    # Runs openBF to patient filae
    #updater.file_openBF(patient_file, "ym - openBF output paciente")

    # Searches optimized parameters
    updater.search_opt("vase1", 0.00001,0.0001, 0.0001, 20)
