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
        np.savetxt(stacked_file, stacked_data, fmt="%.14e")

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
        Horizontally stacks only the 4th column of the partial derivative matrices for each vessel.

        Parameters:
        delta_dict (dict): Dictionary with the deltas used in each parameter.
        output_path (str): Path where the stacked files will be saved.
        """
        parameters = ["h0", "L", "R0"]
        vessels = ["vase1", "vase2", "vase3"]

        # Filtra os parâmetros com delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        for param in parameters:
            if delta_dict[param] == 0:
                print(
                    f"Warning: Variation of the parameter '{param}' is zero. It will be excluded from the Jacobian matrix.")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for vessel in vessels:
            matrices = []

            for param in valid_parameters:
                delta = delta_dict[param]
                file_path = os.path.join(self.openBF_dir, f"partial_deriv_{param}",
                                         f"{vessel}_del_{param}_delta={delta}.last")

                if os.path.exists(file_path):
                    matrix = np.loadtxt(file_path)
                    fourth_column = matrix[:, 3].reshape(-1, 1)  # Seleciona a 4ª coluna e mantém formato 2D
                    matrices.append(fourth_column)
                else:
                    print(f"Error: File not found - {file_path}.")
                    return

            if matrices:
                stacked_matrix = np.hstack(matrices)
                output_file = os.path.join(output_path, f"jacobian_{vessel}_stacked.txt")
                np.savetxt(output_file, stacked_matrix, fmt="%.14e")
                print(f"Stacked matrix saved in: {output_file}")


    def pseudoinverse_matrix(self):
        """ Creates the pseudoinverse_matrix using the Jacobian matrix and saves it in the folder: "jacobians_pseudoinverse"."""
        vessels = ["vase1", "vase2", "vase3"]

        for vessel in vessels:
            file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_{vessel}_stacked.txt")

            if os.path.exists(file_path):
                Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
                JkT_Jk = Jk.T @ Jk # Jacobian transpose times the Jacobian

                # Checks if it is invertible
                # Calculates the conditional number
                cond_number = np.linalg.cond(JkT_Jk)

                print(f"Conditional number ({vessel}): {cond_number:.2e}")

                # Sets a threshold to consider "non-invertible"
                threshold = 1e12

                if cond_number < threshold:
                    print("The matrix is invertible.")
                else:
                    print("Error: The matrix is quasi-singular or non-invertible.")
                    return

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

    def y_til(self, knumber):

        vessels = ["vase1", "vase2", "vase3"]
        Pdk = {vessel: [] for vessel in vessels}

        for vessel in vessels:
            # Loads the data from the patient openBF output - ym
            patient_output = os.path.join(openBF_dir, f"ym - openBF output paciente", f"{vessel}_stacked.last")

            if os.path.exists(patient_output):
                data = np.loadtxt(patient_output)
                patient_data = data[:, 3].reshape(-1, 1) # Pego apenas o 3° nó
            else:
                print(f"Error: Patient output file not found - {patient_output}.")
                return

            # Loads the simulation output corresponding to the initial guess
            yk_output = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}", f"{vessel}_stacked.last")

            if os.path.exists(yk_output):
                data = np.loadtxt(yk_output)
                yk_data = data[:, 3].reshape(-1, 1)  # Pego apenas o 3° nó
            else:
                print(f"Error: Iteration output file not found - {yk_output}.")
                return

            # Loads the Jacobian matrix
            jacobian_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_{vessel}_stacked.txt")

            if os.path.exists(jacobian_path):
                jacobian = np.loadtxt(jacobian_path)
            else:
                print(f"Error: Jacobian matrix file not found - {jacobian_path}.")
                return

            # Creates a vector with the parameter values corresponding to the initial guess
            parameters = ["h0","L","R0"]

            # Loads YAML from k_file
            k_file = os.path.join(openBF_dir, f"problema_inverso - k={knumber}.yaml")
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
            file_dir = os.path.join(openBF_dir, f"Pd{knumber}")
            os.makedirs(file_dir, exist_ok=True)

            # Saves the parameters vector in a file
            Pdk_file = os.path.join(file_dir, f"Pdk_{vessel}.last")
            param_array = np.array(Pdk[vessel]).reshape(-1, 1)
            np.savetxt(Pdk_file, param_array, fmt="%.14e")
            print(f"Parameter vector saved: {Pdk_file}")

            y_til = patient_data - yk_data + (jacobian @ param_array)

            # Where the y_til matrix files will be
            y_til_dir = os.path.join(openBF_dir, "y_til")
            os.makedirs(y_til_dir, exist_ok=True)

            # Saves the y_til matrix in a file
            y_til_file = os.path.join(y_til_dir, f"y_til_{vessel}.last")
            np.savetxt(y_til_file, y_til, fmt="%.14e")
            print(f"y_til matrix saved: {y_til_file}")

    def optimized_parameters(self, knumber):

        vessels = ["vase1", "vase2", "vase3"]

        # Where the matrix of optimized parameters will be
        new_knumber = knumber + 1
        opt_param_dir = os.path.join(openBF_dir, f"optimized_parameters_Pd{new_knumber}")
        os.makedirs(opt_param_dir, exist_ok=True)

        for vessel in vessels:
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

            # Creates the optimized parameters (Pd(k+1)) matrix
            opt_param_data = pseudoinv_data @ y_til_data

            # Saves the optimized parameters matrix in a file
            opt_param_file = os.path.join(opt_param_dir, f"opt_parameters_{vessel}.last")
            np.savetxt(opt_param_file, opt_param_data, fmt="%.14e")
            print(f"Optimized parameters matrix saved: {opt_param_file}")

    def update_yaml_with_optimized_parameters(self, base_yaml_path, param_files_dir, output_yaml_path):
        """
        Atualiza o YAML de entrada usando os parâmetros otimizados salvos em arquivos separados.

        Args:
            base_yaml_path (str): Caminho para o YAML original que será atualizado.
            param_files_dir (str): Diretório onde estão os arquivos 'opt_parameters_vase1.last', etc.
            output_yaml_path (str): Caminho onde o novo YAML atualizado será salvo.
        """

        vessels = ["vase1", "vase2", "vase3"]
        parameters = ["h0", "L", "R0"]

        # Carrega o YAML
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: 'network' key not found in YAML.")
            return

        for vessel in vessels:
            # Carrega o arquivo de parâmetros otimizados
            param_file = os.path.join(param_files_dir, f"opt_parameters_{vessel}.last")

            if os.path.exists(param_file):
                new_params = np.loadtxt(param_file)
            else:
                print(f"Error: Parameter file not found - {param_file}. Skipping {vessel}.")
                continue

            # Garante que new_params é um vetor (não uma matriz)
            new_params = np.atleast_1d(new_params)

            if len(new_params) != len(parameters):
                print(
                    f"Error: Number of parameters mismatch for {vessel}. Expected {len(parameters)}, got {len(new_params)}.")
                continue

            # Atualiza os valores no YAML
            for item in yaml_data["network"]:
                if item.get("label") == vessel:
                    for i, param in enumerate(parameters):
                        item[param] = float(new_params[i])
                    print(f"Updated parameters for {vessel}: {new_params}")
                    break
            else:
                print(f"Warning: Vessel {vessel} not found in YAML.")

        # Salva o novo YAML
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated YAML saved in: {output_yaml_path}")

    def RMSE(self, knumber):
        # Calculates the standard deviation between the patient openBF output and the k openBF output,
        # and between the patient openF and the k + 1 openBF output.

        vessels = ["vase1", "vase2", "vase3"]
        rmse_totais_k = []  # Lista para armazenar o RMSE total de cada vaso
        rmse_totais_kplus = []

        for vessel in vessels:

            # Files paths
            patient_file = os.path.join(openBF_dir, "ym - openBF output paciente", f"{vessel}_stacked.last")
            k_file = os.path.join(openBF_dir, f"y{knumber} - openBF output iteration {knumber}", f"{vessel}_stacked.last")
            kplus_file = os.path.join(openBF_dir, f"y{knumber+1} - openBF output iteration {knumber+1}", f"{vessel}_stacked.last")

            # Checks if files exists
            if not os.path.exists(patient_file):
                print(f"Error: File {patient_file} not found.")
                return
            if not os.path.exists(k_file):
                print(f"Error: File {k_file} not found.")
                return
            if not os.path.exists(kplus_file):
                print(f"Error: File {kplus_file} not found.")
                return

            # Loads stacked files ignoring comments
            patient_data = np.loadtxt(patient_file, comments="#")[:, 1:]  # ignora a 1ª coluna (tempo)
            k_data = np.loadtxt(k_file, comments="#")[:, 1:]
            kplus_data = np.loadtxt(kplus_file, comments="#")[:, 1:]

            def calculate_rmse(y0, y1):
                if y0.shape != y1.shape:
                    raise ValueError("Os arquivos têm formas diferentes!")
                error = y0 - y1
                mse = np.mean(error ** 2, axis=0)  # por coluna
                rmse = np.sqrt(mse)
                return rmse

            ## Calcula RMSE por coluna do k output em relação ao output do paciente
            rmse_colunas_k = calculate_rmse(patient_data, k_data)

            # RMSE médio por vaso
            rmse_total_k = np.mean(rmse_colunas_k)
            rmse_totais_k.append(rmse_total_k)

            ## Calcula RMSE por coluna do k + 1 output em relação ao output do paciente
            rmse_colunas_kplus = calculate_rmse(patient_data, kplus_data)

            # RMSE médio por vaso
            rmse_total_kplus = np.mean(rmse_colunas_kplus)
            rmse_totais_kplus.append(rmse_total_kplus)

        # Cálculo da média dos RMSEs dos 3 vasos para k
        media_rmse_vasos_k = np.mean(rmse_totais_k)
        print(f"\n### FINAL AVERAGE RMSE (k = {knumber} - patient) (3 vessels): {media_rmse_vasos_k:.6f}")

        # Cálculo da média dos RMSEs dos 3 vasos para k+1
        media_rmse_vasos_kplus = np.mean(rmse_totais_kplus)
        print(f"\n### FINAL AVERAGE RMSE (k = {knumber + 1} - patient) (3 vessels): {media_rmse_vasos_kplus:.6f}")

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
        variables = ["A", "P", "Q", "u"]

        # Stack openBF outputs for each vessel individually
        for vessel in vessels:
            self.stack_last_files(vessel, variables, file_dir)

        # Plots the simulation output graphs and saves them
        self.plot_openBF(file_dir)


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

    def iteration(self, knumber, vase, add_h0, add_L, add_R0):
        """Creates the Jacobian pseudoinverse matrix considering the increments specified for each parameter,
        multiplies it to the y_til matrix and generates the optimized parameters."""
        add_values = {"h0": add_h0, "L": add_L, "R0": add_R0}

        if knumber == 0:  # Runs openBF to k-iteration YAML file only if k = 0
            k_yaml_file = f"C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso - k={knumber}.yaml"
            self.file_openBF(k_yaml_file, f"y{knumber} - openBF output iteration {knumber}")

        for parameter in add_values:
            self.updated_openBF(knumber, vase, parameter, add_values[parameter])

        # Path to the Jacobian matrices directory
        output_path = os.path.join(openBF_dir, f"jacobians")
        self.stack_partial_derivatives(add_values, output_path)

        # Creates the pseudoinverse matrix
        self.pseudoinverse_matrix()

        # Creates the y_til matrix
        self.y_til(knumber)

        # Creates the optimized parameters matrix
        self.optimized_parameters(knumber)

        # Updates YAML with optimized parameters
        base_yaml_path = f"C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso - k={knumber}.yaml"
        opt_param_files_dir = f"C:/Users/User/Documents/problema_inverso_results_openbf/optimized_parameters_Pd{knumber+1}"
        opt_output_yaml_path = f"C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso - k={knumber+1}.yaml"
        self.update_yaml_with_optimized_parameters(base_yaml_path, opt_param_files_dir, opt_output_yaml_path)

        # Runs openBF to the new/optimized yaml file
        self.file_openBF(opt_output_yaml_path, f"y{knumber+1} - openBF output iteration {knumber+1}")

        # Calculates the RMSE for the old and new openBF output
        self.RMSE(knumber)


    def search_opt(self, vase, add_h0, add_L, add_R0):

        # Starts cronometer
        start = time.time()

        # Runs iteration for knumber from 0 to 3
        for knumber in range(0, 4):
            self.iteration(knumber, vase, add_h0, add_L, add_R0)

        # Ends cronometer and prints time
        end = time.time()
        print(f"Elapsed time: {end - start:.3f} seconds.")




# Application
if __name__ == "__main__":

    patient_file = "C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso - Paciente.yaml"
    k0_file = "C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso - k=0.yaml"
    openBF_dir = "C:/Users/User/Documents/problema_inverso_results_openbf"

    updater = OPENBF_Jacobian(patient_file, k0_file, openBF_dir)

    # Runs openBF to patient file
    #updater.file_openBF(patient_file, "ym - Output paciente")

    # Iteration
    #updater.iteration(0,"vase1", 0.0001,0.001, 0.001)

    # teste
    updater.search_opt("vase1", 0.0001,0.001, 0.001)
