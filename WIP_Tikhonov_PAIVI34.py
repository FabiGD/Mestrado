import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time
import sys
from scipy.signal import savgol_filter


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
    def __init__(self, openBF_dir, inlet_dat, patient_yaml, k0_yaml, kstar_txt):
        self.openBF_dir = openBF_dir
        self.inlet_file = inlet_dat # Is not used explicitly, but needs to exist
        self.patient_file = patient_yaml
        self.k0_file = k0_yaml 
        self.kstar_file = kstar_txt


    def add_noise(self, vessel):
        """ Adds noise to pacient openBF output."""

        # Loads the data from the patient openBF output
        patient_output = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_stacked.last")
        if not os.path.exists(patient_output):
            raise SystemExit(f"Error: Patient output file not found - {patient_output}")
        patient_data = np.loadtxt(patient_output, comments="#")[:, 3] # Only takes the 3rd knot

        # Standard deviations 
        pressure_std = 266.64 # Pressure standard deviation [Pa]

        velocity_mean = np.mean(patient_data[-100:])  
        velocity_std = 0.03 * velocity_mean # Velocity standard deviation [m/s]

        # Adds noise to the first 100 lines of patient_data (standard deviation = 266.64 Pa)
        patient_withnoise = patient_data.copy() # Copies patient_data
        patient_withnoise[:100] = patient_data[:100] + pressure_std * np.random.randn(100)

        # Adds noise to the last 100 lines of patient_data (standard deviation = 3% of the mean)
        patient_withnoise[-100:] = patient_data[-100:] + velocity_std * np.random.randn(100)

        # Saves the result
        file_path = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_withnoise.last")
        np.savetxt(file_path, patient_withnoise, fmt="%.14e")
        print(f"Patient openBF output with noise saved: {file_path}")

    def equal_parameters(self, yaml_file):
        """
        Checks if parameters R1 and Cc are equal in vessels 2 and 3.
        Returns True if both are equal, otherwise False.
        """

        # Checks if yaml file exists
        if not os.path.exists(yaml_file):
            raise SystemExit(f"Error: File {yaml_file} not found. Execution stopped.")

        # Loads yaml
        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the yaml file.")
            return False

        # Initializes values
        params_v2, params_v3 = {}, {}

        for item in yaml_data["network"]:
            if item.get("label") == "vessel2":
                params_v2["R1"] = item.get("R1")
                params_v2["Cc"] = item.get("Cc")
            elif item.get("label") == "vessel3":
                params_v3["R1"] = item.get("R1")
                params_v3["Cc"] = item.get("Cc")

        # Checks if values were found
        if not params_v2 or not params_v3:
            print(f"Error: Vessel2 or Vessel3 not found in {yaml_file}.")
            return False

        # Returns boolean comparison
        return params_v2["R1"] == params_v3["R1"] and params_v2["Cc"] == params_v3["Cc"]


    def update_yaml(self, knumber, vessel, parameter, add_value):
        """       
        Updates the parameter in the specified vessel.
        """

        self.add_value = add_value
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}.yaml")

        # Loads YAML from k_file
        if knumber == 0:
            k_file = os.path.join(openBF_dir, self.k0_file)
        else:
            k_file = os.path.join(openBF_dir, f"inverse_problem_k={knumber}.yaml")
        with open(k_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the YAML.")
            return

        found = False  # Flag to know if the vessel was found

        for item in yaml_data["network"]:
            print(f" Checking vessel: {item.get('label')}")  # Debug print
            if item.get("label") == vessel:
                if parameter in item:
                    item[parameter] = item[parameter] + add_value
                    print(f"Parameter '{parameter}' of vessel '{vessel}' updated to: {item[parameter]}")
                    found = True
                else:
                    print(f"Error: Parameter '{parameter}' not found in vessel '{vessel}'.")
                break

        if not found:
            print(f"Error: Vessel '{vessel}' not found in YAML.")

        # Saves the updated YAML to the updated_file
        with open(updated_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        print(f"Updated file saved in: {updated_file}")


    def update_yaml_tied(self, knumber, parameter, add_value):
        """ 
        Simultaneously updates the parameter (e.g., R1 or Cc) in vessels 2 and 3.
        """

        self.add_value = add_value
        updated_file = os.path.join(openBF_dir, f"updated_WK.yaml")
        vessels = ["vessel2", "vessel3"]

        # Loads YAML from k_file
        if knumber == 0:
            k_file = os.path.join(openBF_dir, self.k0_file)
        else:
            k_file = os.path.join(openBF_dir, f"inverse_problem_k={knumber}.yaml")
        with open(k_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the YAML.")
            return

        for vessel in vessels:
            found = False
            for item in yaml_data["network"]:
                if item.get("label") == vessel:
                    if parameter in item:
                        item[parameter] = item[parameter] + add_value
                        print(f"Parameter '{parameter}' of vessel '{vessel}' updated to: {item[parameter]}")
                        found = True
                    else:
                        print(f"Error: Parameter '{parameter}' not found in vessel '{vessel}'.")
                    break
            if not found:
                print(f"Error: Vessel '{vessel}' not found in YAML.")

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

    def stack_last_files(self, file_list, data_dir, output_ID):
        """
        Vertically stacks the .last files provided in the order specified in file_list
        and saves the results (stacked and spectral norm).

        Parameters
        ----------
        file_list: list of str
        List of .last file names (without path).
        The order of the list will be the stacking order.
        data_dir: str
        Path to the directory where the files are located.
        vessel_name: str, optional
        Prefix for the output file name (default="stacked").
        """

        data_list = []

        for file_name in file_list:
            file_path = os.path.join(data_dir, file_name)

            # Checks if file exists
            if not os.path.exists(file_path):
                raise RuntimeError(
                    f"Error: File not found: {file_path}. Check if the simulation ran correctly."
                )
            
            # Reads data from .last file
            data = np.loadtxt(file_path)
            data_list.append(data)

        # Stacks vertically
        stacked_data = np.vstack(data_list)

        # Calculates the spectral norm (largest singular value)
        spectral_norm = np.linalg.norm(stacked_data, 2)

        # Saves the stacked file
        stacked_file = os.path.join(data_dir, f"{output_ID}_stacked.last")
        np.savetxt(stacked_file, stacked_data, fmt="%.14e")

        # Saves the spectral norm
        spectral_file = os.path.join(data_dir, f"{output_ID}_spectral_norm.last")
        np.savetxt(spectral_file, np.array([spectral_norm]), fmt="%.14e")

        print(f"Saved file: {stacked_file}")
        print(f"Saved file: {spectral_file}")


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

        base_file_path = os.path.join(base_dir, f"{vessel}_stacked.last") 
        updated_file_path = os.path.join(updated_dir, f"{vessel}_stacked.last")
        del_file_path = os.path.join(del_dir, f"{vessel}_del_{parameter}_delta={delta_value:.0e}.last")

        # Checks if both files exist
        if not os.path.exists(base_file_path) or not os.path.exists(updated_file_path):
            raise SystemExit(f"Error: Files for {vessel} not found. Execution stopped.")

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

    
    def tied_partial_deriv_files(self, vessel, base_dir, updated_dir, del_dir, parameter):
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

        base_file_path = os.path.join(base_dir, f"{vessel}_stacked.last") 
        updated_file_path = os.path.join(updated_dir, f"{vessel}_stacked.last")
        del_file_path = os.path.join(del_dir, f"{vessel}_del_{parameter}_delta={delta_value:.0e}.last")

        # Checks if both files exist
        if not os.path.exists(base_file_path) or not os.path.exists(updated_file_path):
            raise SystemExit(f"Error: Files for {vessel} not found. Execution stopped.")

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


    def stack_partial_derivatives(self, knumber, vessel, delta_dict, output_path):
        """
        Horizontally stacks only the 4th column of the partial derivative matrices for each vessel.

        Parameters:
        delta_dict (dict): Dictionary with the deltas used in each parameter.
        output_path (str): Path where the stacked files will be saved.
        """
        parameters = ["h0", "L", "R0", "Rp", "Rd", "E", "R1", "R2", "Cc"]

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
                                        f"{vessel}_del_{param}_delta={delta:.0e}.last")

            if os.path.exists(file_path):
                matrix = np.loadtxt(file_path)
                fourth_column = matrix[:, 3].reshape(-1, 1)  # Selects the 4th column and keeps 2D format
                matrices.append(fourth_column)
            else:
                raise SystemExit(f"Error: File not found - {file_path}. Execution stopped.")

        if matrices:
            stacked_matrix = np.column_stack(matrices)
            output_file = os.path.join(output_path, f"jacobian_k={knumber}_{vessel}_stacked.txt")
            np.savetxt(output_file, stacked_matrix, fmt="%.14e")
            print(f"Stacked matrix saved in: {output_file}")


    def tied_stack_partial_derivatives(self, knumber, delta_dict, output_path):
        """
        Vertically stacks the 4th column of the partial derivative matrices for each vessel;
        Horizontally stacks only the 4th column of the partial derivative matrices for each parameter.

        """
        vessels = ["vessel2", "vessel3"]
        parameters = ["R1", "Cc"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        for param in parameters:
            if delta_dict[param] == 0:
                print(
                    f"Warning: Variation of the parameter '{param}' is zero. It will be excluded from the Jacobian matrix.")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # List to save the Jacobian columns
            jacobian_columns = []

            # For each parameter, stack the derivatives of the vessels
            for param in valid_parameters:
                col_blocks = []
                delta = delta_dict[param]

                for vessel in vessels:
                    file_path = os.path.join(
                        openBF_dir, "partial_deriv_WK",
                        f"{vessel}_del_{param}_delta={delta:.0e}.last"
                    )

                    if not os.path.exists(file_path):
                        raise SystemExit(f"Error: File not found - {file_path}")

                    matrix = np.loadtxt(file_path)
                    fourth_column = matrix[:, 3].reshape(-1, 1)
                    col_blocks.append(fourth_column)

                # Stacks the vessels vertically to form the Jacobian matrix column
                jacobian_column = np.vstack(col_blocks)
                jacobian_columns.append(jacobian_column)

            # Horizontally stacks all the columns to form the complete Jacobian matrix
            if jacobian_columns:
                J = np.hstack(jacobian_columns)
                output_file = os.path.join(output_path, f"jacobian_k={knumber}_vessels2and3.txt")
                np.savetxt(output_file, J, fmt="%.14e")
                print(f"Jacobian matrix saved in: {output_file}")


    def A_matrix(self, ID, beta_opt, vessel, knumber):
        """ Creates the A_matrix using the Jacobian matrix and saves it in the folder: "A_matrix_beta={beta:.0e}"."""

        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_{vessel}_stacked.txt")

        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
            if Jk.ndim == 1:
                Jk = Jk.reshape(-1, 1)

            # Required files paths
            yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"{vessel}_stacked.last")
            k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
            
            # Checks if files exist
            required_files = [
                yk_file,
                k0_param_file
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")

            # Loads files ignoring comments
            yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
            k0_data = np.atleast_1d(np.loadtxt(k0_param_file, comments="#"))

            # Tikhonov Regularization matrices
            z = yk_data
            W1 = np.diag(1 / (z**2)) # Weighting matrix

            P = k0_data
            W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

            # Regularizing the solution of the LS-problem
            JkT_W1_Jk = Jk.T @ W1 @ Jk # Jacobian transpose times the Jacobian with W1 matrix
            A_matrix = JkT_W1_Jk + beta_opt**2 * W2

            # Saves A matrix 
            output_dir = os.path.join(openBF_dir, f"A_matrix_ID={ID}")
            os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it does not exist

            output_file = os.path.join(output_dir, f"A_matrix_{vessel}.txt")
            np.savetxt(output_file, A_matrix, fmt="%.14e")
            print(f"A matrix saved in: {output_file}")

        else:
            raise SystemExit(f"Error: File not found - {file_path}. Execution stopped.")
        

    def tied_A_matrix(self, ID, beta_opt, knumber):
        """ Creates the A_matrix using the Jacobian matrix and saves it in the folder: "A_matrix_beta={beta:.0e}". 
            Used in tied_iteration."""

        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_vessels2and3.txt")

        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
            if Jk.ndim == 1:
                Jk = Jk.reshape(-1, 1)

            # Required files paths
            yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"vessels2and3_stacked.last")
            k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_vessel2.last") 
            
            # Checks if files exist
            required_files = [
                yk_file,
                k0_param_file
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")

            # Loads files ignoring comments
            yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
            k0_data = np.atleast_1d(np.loadtxt(k0_param_file, comments="#"))

            # Tikhonov Regularization matrices
            z = yk_data
            W1 = np.diag(1 / (z**2)) # Weighting matrix

            P = k0_data
            W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

            # Regularizing the solution of the LS-problem
            JkT_W1_Jk = Jk.T @ W1 @ Jk # Jacobian transpose times the Jacobian with W1 matrix
            A_matrix = JkT_W1_Jk + beta_opt**2 * W2

            # Saves A matrix 
            output_dir = os.path.join(openBF_dir, f"A_matrix_ID={ID}")
            os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it does not exist

            output_file = os.path.join(output_dir, f"A_matrix_WK.txt")
            np.savetxt(output_file, A_matrix, fmt="%.14e")
            print(f"A matrix saved in: {output_file}")

        else:
            raise SystemExit(f"Error: File not found - {file_path}. Execution stopped.")


    def B_matrix(self, ID, beta_opt, vessel, knumber):

        # Path to the jacobian matrix file
        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_{vessel}_stacked.txt")
        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
        else:
            raise SystemExit(f"Error: Jacobian matrix file not found - {file_path}")

        # Loads the data from the patient openBF output
        patient_output = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_withnoise.last")
        if not os.path.exists(patient_output):
            raise SystemExit(f"Error: Patient output file not found - {patient_output}")
        patient_data = np.loadtxt(patient_output, comments="#")

        # Loads the simulation output corresponding to the guess
        yk_output = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"{vessel}_stacked.last")
        if not os.path.exists(yk_output):
            raise SystemExit(f"Error: Output file for iteration {knumber} not found - {yk_output}")
        yk_data = np.loadtxt(yk_output, comments="#")[:, 3] # Only takes the 3rd knot

        # Residual matrix
        R_matrix = patient_data - yk_data

        # Loads the iteration k paramaters and initial parameters (Pk and P0)
        k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
        if knumber == 0:
            k_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
        else:
            k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_{vessel}.last")


        if os.path.exists(k0_param_file):
            k0_param_data = np.loadtxt(k0_param_file)
        else:
            raise SystemExit(f"Error: Initial parameters file not found - {k0_param_file}")
        
        if os.path.exists(k_param_file):
            k_param_data = np.loadtxt(k_param_file)
        else:
            raise SystemExit(f"Error: Iteration parameters file not found - {k_param_file}")
        
        # Loads the star paramaters (reliable guess)
        kstar_param_file = self.kstar_file
        if os.path.exists(kstar_param_file):
            kstar_data = np.loadtxt(kstar_param_file)
        else:
            raise SystemExit(f"Error: Star paramaters (reliable guess) file not found - {kstar_param_file}")

        # Tikhonov Regularization matrices
        z = yk_data
        W1 = np.diag(1 / (z**2)) # Weighting matrix

        P = k0_param_data
        W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix
        
        # Creates the regularized B matriz
        B = Jk.T @ W1 @ R_matrix - beta_opt**2 * W2 @ (k_param_data - kstar_data)

        # Where the B matrix files will be
        B_dir = os.path.join(openBF_dir, "B_matrix")
        os.makedirs(B_dir, exist_ok=True)

        # Saves the B matrix in a file
        B_file = os.path.join(B_dir, f"B_matrix_{vessel}_ID={ID}.last")
        np.savetxt(B_file, np.atleast_1d(B), fmt="%.14e")
        print(f"B matrix saved: {B_file}")


    def tied_B_matrix(self, ID, beta_opt, knumber):

        # Path to the jacobian matrix file
        file_path = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_vessels2and3.txt")
        if os.path.exists(file_path):
            Jk = np.loadtxt(file_path) # Loads the Jacobian matrix
        else:
            raise SystemExit(f"Error: Jacobian matrix file not found - {file_path}")

        # Loads the data from the stacked patient openBF output 
        patient_output = os.path.join(openBF_dir, "ym_openBF_patient_output", f"vessels2and3_withnoise_stacked.last")
        if not os.path.exists(patient_output):
            raise SystemExit(f"Error: Patient output file not found - {patient_output}")
        patient_data = np.loadtxt(patient_output, comments="#")

        # Loads the stacked simulation output
        yk_output = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"vessels2and3_stacked.last")
        if not os.path.exists(yk_output):
            raise SystemExit(f"Error: Output file for iteration {knumber} not found - {yk_output}")
        yk_data = np.loadtxt(yk_output, comments="#")[:, 3] # Only takes the 3rd knot

        # Residual matrix
        R_matrix = patient_data - yk_data

        # Loads the iteration k paramaters and initial parameters (Pk and P0)
        k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_vessel2.last")
        if knumber == 0:
            k_param_file = k0_param_file
        else:
            k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_vessels2and3.last")


        if os.path.exists(k0_param_file):
            k0_param_data = np.loadtxt(k0_param_file)
        else:
            raise SystemExit(f"Error: Initial parameters file not found - {k0_param_file}")
        
        if os.path.exists(k_param_file):
            k_param_data = np.loadtxt(k_param_file)
        else:
            raise SystemExit(f"Error: Iteration parameters file not found - {k_param_file}")
        
        # Loads the star paramaters (reliable guess)
        kstar_param_file = self.kstar_file
        if os.path.exists(kstar_param_file):
            kstar_data = np.loadtxt(kstar_param_file)
        else:
            raise SystemExit(f"Error: Star paramaters (reliable guess) file not found - {kstar_param_file}")

        # Tikhonov Regularization matrices
        z = yk_data
        W1 = np.diag(1 / (z**2)) # Weighting matrix

        P = k0_param_data
        W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix
        
        # Creates the regularized B matriz
        B = Jk.T @ W1 @ R_matrix - beta_opt**2 * W2 @ (k_param_data - kstar_data)

        # Where the B matrix files will be
        B_dir = os.path.join(openBF_dir, "B_matrix")
        os.makedirs(B_dir, exist_ok=True)

        # Saves the B matrix in a file
        B_file = os.path.join(B_dir, f"B_matrix_WK_ID={ID}.last")
        np.savetxt(B_file, np.atleast_1d(B), fmt="%.14e")
        print(f"B matrix saved: {B_file}")


    def Pk(self, vessel, delta_dict, param_directory, yaml_file):
        """Loads the parameters of a yaml file and saves it in a directory."""

        Pk = {vessel: []}

        # Creates a vector with the parameter values corresponding to the guess
        parameters = ["h0", "L", "R0", "Rp", "Rd", "E", "R1", "R2", "Cc"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        # Loads YAML from k_file
        k_file = os.path.join(openBF_dir, yaml_file)
        with open(k_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: The 'network' key was not found in the YAML.")
            return

        found = False  # Flag to know if the vessel was found

        for item in yaml_data["network"]:
            if item.get("label") == vessel:
                for parameter in valid_parameters:
                    if parameter in item:
                        value = item[parameter]
                        Pk[vessel].append(value)
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
        Pk_file = os.path.join(file_dir, f"Pk_{vessel}.last")
        param_array = np.array(Pk[vessel]).reshape(-1, 1)
        np.savetxt(Pk_file, param_array, fmt="%.14e")
        print(f"Parameter vector saved: {Pk_file}")


    def optimized_parameters(self, ID, vessel, alpha, beta_opt, knumber):
        """ Obtains the optimized parameters Pk+1 and saves them in a file.
        
        Parameters:
            alpha (int): Alpha is the sub-relaxation factor for the Newton step."""

        # Where the matrix of optimized parameters will be
        new_knumber = knumber + 1
        opt_param_dir = os.path.join(openBF_dir, f"optimized_parameters_P{new_knumber}")
        os.makedirs(opt_param_dir, exist_ok=True)

        # Loads the parameters of the initial guess (Pk)
        if knumber == 0:
            param_path = os.path.join(openBF_dir, f"P{knumber}", f"Pk_{vessel}.last")
        else:
            param_path = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}",
                                        f"Pk_{vessel}.last")

        if os.path.exists(param_path):
            param_data = np.loadtxt(param_path).ravel() # Ensures vector (n_params,)
        else:
            raise SystemExit(f"Error: Pk matrix file not found - {param_path}. Execution stopped.")

        # Loads the data from the A matrix
        A_matrix_path = os.path.join(openBF_dir, f"A_matrix_ID={ID}", f"A_matrix_{vessel}.txt")

        if os.path.exists(A_matrix_path):
            A_data = np.loadtxt(A_matrix_path)
            if A_data.ndim == 0:
                A_data = np.array([[A_data]]) # scalar becomes 1x1 matrix
        else:
            raise SystemExit(f"Error: A matrix file not found - {A_matrix_path}. Execution stopped.")

        # Loads the data from the B matrix
        B_matrix_path = os.path.join(openBF_dir, "B_matrix", f"B_matrix_{vessel}_ID={ID}.last")

        if os.path.exists(B_matrix_path):
            B_data = np.loadtxt(B_matrix_path)
            # Adjusting the dimensions to use np.lingalg.solve
            if B_data.ndim == 0:
                B_data = np.array([B_data]) # scalar becomes 1D vector
            B_data = B_data.reshape(-1, 1)

        else:
            raise SystemExit(f"Error: B matrix file not found - {B_matrix_path}. Execution stopped.")


        # Creates the optimized parameters (P(k+1)) matrix
        deltaP_matrix = np.linalg.solve(A_data,B_data)
        print("Shape of deltaP is:", deltaP_matrix.shape)
        deltaP_matrix = deltaP_matrix.ravel()   # Ensures vector (n_params,)
        print("Fixed shape of deltaP is:", deltaP_matrix.shape)

        print("Shape param_data:", param_data.shape)
        opt_param_data = param_data + alpha * deltaP_matrix
        print(f"Optimized parameters (Pk+1): {opt_param_data}, shape: {opt_param_data.shape}")

        # Checks rank for A_data
        threshold = 1e-12 # threshold to consider a number equivalent to zero
        u_A, s_A, vh_A = np.linalg.svd(A_data)
        rank_A = np.sum(s_A > threshold)

        # Checks condition number for A_data @ deltaP_matrix
        threshold_cond = 1e6 # threshold to consider ill-conditioned
        cond_A = np.linalg.norm(A_data) * np.linalg.norm(np.linalg.inv(A_data))
        cond_AdP = np.linalg.norm(A_data) * (np.linalg.norm(deltaP_matrix)/np.linalg.norm(A_data @ deltaP_matrix))

        # Prints and saves in a file the rank, condition number and the status of the matrices
        print_path = os.path.join(openBF_dir, f"A_matrix_ID={ID}", f"condition_{vessel}_k{knumber}.txt")
        os.makedirs(os.path.dirname(print_path), exist_ok=True)

        with open(print_path, "w") as log:
            msg_rank = f"A-matrix rank: {rank_A:.2e}"
            rank_status = (
                f"Error: The A-matrix is rank-deficient (non-invertible)."
                if rank_A < min(A_data.shape)
                else f"The A-matrix is full-rank (invertible)."
            )
            msg_cond1 = f"A-matrix condition number: {cond_A:.2e}"
            cond_status1 = (
                f"A-matrix is well-conditioned."
                if cond_A < threshold_cond
                else f"Warning: A-matrix is ill-conditioned (Condition number >= {threshold_cond:.0e})."
            )
            msg_cond2 = f"(A-matrix @ delta P matrix) condition number: {cond_AdP:.2e}"
            cond_status2 = (
                f"(A-matrix @ delta P matrix) is well-conditioned."
                if cond_AdP < threshold_cond
                else f"Warning: The multiplication (A-matrix @ delta P matrix) is ill-conditioned (Condition number >= {threshold_cond:.0e})."
            )
            
            print(msg_rank)
            print(rank_status)
            print(msg_cond1)
            print(cond_status1)
            print(msg_cond2)
            print(cond_status2)

            log.write(msg_rank + "\n")
            log.write(rank_status + "\n")
            log.write(msg_cond1 + "\n")
            log.write(cond_status1 + "\n")
            log.write(msg_cond2 + "\n")
            log.write(cond_status2 + "\n")
            log.write("\n")

            print(f"beta = {beta_opt:.0e}")
            log.write(f"beta = {beta_opt:.0e} \n")


        # Saves the optimized parameters matrix in a file
        opt_param_file = os.path.join(opt_param_dir, f"Pk_{vessel}.last")
        np.savetxt(opt_param_file, opt_param_data, fmt="%.14e")
        print(f"Optimized parameters matrix saved: {opt_param_file}")

        # Checking optimized parameters: prevents negative values
        if np.any(opt_param_data < 0):
            raise ValueError(
                f"Invalid parameter: Negative value detected in {opt_param_data}.\n"
                f"Check the value of alpha or the input data."
            )


    def tied_optimized_parameters(self, ID, alpha, beta_opt, knumber):
        """ Obtains the optimized parameters Pk+1 and saves them in a file. Used in tied iteration."""

        # Where the matrix of optimized parameters will be
        new_knumber = knumber + 1
        opt_param_dir = os.path.join(openBF_dir, f"optimized_parameters_P{new_knumber}")
        os.makedirs(opt_param_dir, exist_ok=True)

        # Loads the parameters of the initial guess (Pk)
        if knumber == 0:
            param_path = os.path.join(openBF_dir, f"P{knumber}", f"Pk_vessel2.last")
        else:
            param_path = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}",
                                        f"Pk_vessels2and3.last")

        if os.path.exists(param_path):
            param_data = np.loadtxt(param_path).ravel() # Ensures vector (n_params,)
        else:
            raise SystemExit(f"Error: Pk matrix file not found - {param_path}. Execution stopped.")

        # Loads the data from the A matrix
        A_matrix_path = os.path.join(openBF_dir, f"A_matrix_ID={ID}", f"A_matrix_WK.txt")

        if os.path.exists(A_matrix_path):
            A_data = np.loadtxt(A_matrix_path)
            if A_data.ndim == 0:
                A_data = np.array([[A_data]]) # scalar becomes 1x1 matrix
        else:
            raise SystemExit(f"Error: A matrix file not found - {A_matrix_path}. Execution stopped.")

        # Loads the data from the B matrix
        B_matrix_path = os.path.join(openBF_dir, "B_matrix", f"B_matrix_WK_ID={ID}.last")

        if os.path.exists(B_matrix_path):
            B_data = np.loadtxt(B_matrix_path)
            # Adjusting the dimensions to use np.lingalg.solve
            if B_data.ndim == 0:
                B_data = np.array([B_data]) # scalar becomes 1D vector
            B_data = B_data.reshape(-1, 1)

        else:
            raise SystemExit(f"Error: B matrix file not found - {B_matrix_path}. Execution stopped.")


        # Creates the optimized parameters (P(k+1)) matrix
        deltaP_matrix = np.linalg.solve(A_data,B_data)
        print("Shape of deltaP is:", deltaP_matrix.shape)
        deltaP_matrix = deltaP_matrix.ravel()   # Ensures vector (n_params,)
        print("Fixed shape of deltaP is:", deltaP_matrix.shape)

        print("Shape param_data:", param_data.shape)
        opt_param_data = param_data + alpha * deltaP_matrix
        print(f"Optimized parameters (Pk+1): {opt_param_data}, shape: {opt_param_data.shape}")

        # Checks rank for A_data
        threshold = 1e-12 # threshold to consider a number equivalent to zero
        u_A, s_A, vh_A = np.linalg.svd(A_data)
        rank_A = np.sum(s_A > threshold)

        # Checks condition number for A_data @ deltaP_matrix
        threshold_cond = 1e6 # threshold to consider ill-conditioned
        cond_A = np.linalg.norm(A_data) * np.linalg.norm(np.linalg.inv(A_data))
        cond_AdP = np.linalg.norm(A_data) * (np.linalg.norm(deltaP_matrix)/np.linalg.norm(A_data @ deltaP_matrix))

        # Prints and saves in a file the rank, condition number and the status of the matrices
        print_path = os.path.join(openBF_dir, f"A_matrix_ID={ID}", f"condition_WK_k{knumber}.txt")
        os.makedirs(os.path.dirname(print_path), exist_ok=True)

        with open(print_path, "w") as log:
            msg_rank = f"A-matrix rank: {rank_A:.2e}"
            rank_status = (
                f"Error: The A-matrix is rank-deficient (non-invertible)."
                if rank_A < min(A_data.shape)
                else f"The A-matrix is full-rank (invertible)."
            )
            msg_cond1 = f"A-matrix condition number: {cond_A:.2e}"
            cond_status1 = (
                f"A-matrix is well-conditioned."
                if cond_A < threshold_cond
                else f"Warning: A-matrix is ill-conditioned (Condition number >= {threshold_cond:.0e})."
            )
            msg_cond2 = f"(A-matrix @ delta P matrix) condition number: {cond_AdP:.2e}"
            cond_status2 = (
                f"(A-matrix @ delta P matrix) is well-conditioned."
                if cond_AdP < threshold_cond
                else f"Warning: The multiplication (A-matrix @ delta P matrix) is ill-conditioned (Condition number >= {threshold_cond:.0e})."
            )
            
            print(msg_rank)
            print(rank_status)
            print(msg_cond1)
            print(cond_status1)
            print(msg_cond2)
            print(cond_status2)

            log.write(msg_rank + "\n")
            log.write(rank_status + "\n")
            log.write(msg_cond1 + "\n")
            log.write(cond_status1 + "\n")
            log.write(msg_cond2 + "\n")
            log.write(cond_status2 + "\n")
            log.write("\n")

            print(f"beta = {beta_opt:.0e}")
            log.write(f"beta = {beta_opt:.0e} \n")


        # Saves the optimized parameters matrix in a file
        opt_param_file = os.path.join(opt_param_dir, f"Pk_vessels2and3.last")
        np.savetxt(opt_param_file, opt_param_data, fmt="%.14e")
        print(f"Optimized parameters matrix saved: {opt_param_file}")

        # Checking optimized parameters: prevents negative values
        if np.any(opt_param_data < 0):
            raise ValueError(
                f"Invalid parameter: Negative value detected in {opt_param_data}.\n"
                f"Check the value of alpha or the input data."
            )


    def update_yaml_with_optimized_parameters(self, vessel, delta_dict, base_yaml_path, param_files_dir, output_yaml_path):
        # Updates the input YAML using the optimized parameters saved in separate files.

        parameters = ["h0", "L", "R0", "Rp", "Rd", "E", "R1", "R2", "Cc"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        # Loads the YAML file
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: 'network' key not found in YAML.")
            return

        # Loads the file with the optimized parameters
        param_file = os.path.join(param_files_dir, f"Pk_{vessel}.last")

        if os.path.exists(param_file):
            new_params = np.loadtxt(param_file)
        else:
            raise SystemExit(f"Error: Parameter file '{param_file}' not found. Execution stopped.")

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


    def tied_update_yaml_with_optimized_parameters(self, delta_dict, base_yaml_path, param_files_dir, output_yaml_path):
        # Updates the input YAML using the optimized parameters saved in separate files.

        parameters = ["R1", "Cc"]

        # Filters parameters with delta != 0
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        # Loads the YAML file
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "network" not in yaml_data:
            print("Error: 'network' key not found in YAML.")
            return

        # Loads the file with the optimized parameters
        param_file = os.path.join(param_files_dir, f"Pk_vessels2and3.last")

        if os.path.exists(param_file):
            new_params = np.loadtxt(param_file)
        else:
            raise SystemExit(f"Error: Parameter file '{param_file}' not found. Execution stopped.")

        # Ensures that new_params is a vector (not an array)
        new_params = np.atleast_1d(new_params)

        if len(new_params) != len(valid_parameters):
            print(
                f"Error: Number of parameters mismatch for Windkessel model. Expected {len(valid_parameters)}, got {len(new_params)}.")

        # Updates the YAML values
        vessels = ["vessel2", "vessel3"]
        for vessel_update in vessels:
            for item in yaml_data["network"]:
                if item.get("label") == vessel_update:
                    for i, param in enumerate(valid_parameters):
                        item[param] = float(new_params[i])
                    print(f"Updated parameters for {vessel_update}: {new_params}")
                    break
            else:
                print(f"Warning: Vessel {vessel_update} not found in YAML.")

        # Saves the new YAML file
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated YAML saved in: {output_yaml_path}")


    def plot_error(self, ID, vessel, beta_method, knumber_max):
        # Plots the total error vs. iteration

        plt.close('all')
        residual_error_append = []
        solution_error_append = []
        

        for knumber in range(0, knumber_max + 1):

            kplus = knumber + 1

            # Files paths
            patient_file = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_withnoise.last")
            yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"{vessel}_stacked.last")
            Jk_file = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_{vessel}_stacked.txt")
            kplus_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{kplus}", f"Pk_{vessel}.last")
            k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
            if knumber == 0:
                k_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
            else:
                k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_{vessel}.last")
            kstar_param_file = self.kstar_file

            # Checks if files exist
            required_files = [
                patient_file,
                yk_file,
                Jk_file,
                kplus_param_file,
                k0_param_file,
                k_param_file,
                kstar_param_file
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")


            # Loads files ignoring comments
            patient_data = np.loadtxt(patient_file, comments="#")
            yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
            Jk_data = np.loadtxt(Jk_file, comments="#")
            kplus_data = np.loadtxt(kplus_param_file, comments="#")
            k0_data = np.loadtxt(k0_param_file, comments="#") # verificar se apago
            k_data = np.loadtxt(k_param_file, comments="#")
            kstar_data = np.loadtxt(kstar_param_file, comments="#")

            # Corrects Jk dimension 
            if Jk_data.ndim == 1:
                Jk_data = Jk_data.reshape(-1, 1) # (200,) -> (200,1)

            # Creates deltaP matrix
            deltaP_matrix = kplus_data - k_data

            # Tikhonov Regularization matrices
            z = yk_data
            W1 = np.diag(1 / (z**2)) # Weighting matrix

            P = k0_data
            W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

            # Calculates the residual norm
            res = patient_data - yk_data - Jk_data @ deltaP_matrix
            res_error = 0.5 * (res.T @ W1 @ res)

            # Stores it to plot
            residual_error_append.append(res_error)

            # Finds optimal beta
            if beta_method == "L_curve":
                beta_opt = self.Lcurve_dict.get(knumber, list(self.Lcurve_dict.values())[-1])
            if beta_method == "Morozov":
                beta_opt = self.Morozov_dict.get(knumber, list(self.Morozov_dict.values())[-1])
            else:
                raise ValueError(f"Unknown beta_method: {beta_method}. Please insert 'L_curve' or 'Morozov'.")


            # Calculates the solution norm
            sol = np.atleast_1d((kplus_data - kstar_data)) 
            sol_error = 0.5 * beta_opt**2 * (sol.T @ W2 @ sol)

            # Stores it to plot
            solution_error_append.append(sol_error)


        # Iterations: 0 to knumber_max
        iterations = np.arange(0, knumber_max + 1)

        # Plots directory
        plots_dir = os.path.join(openBF_dir, f"iteration_plots_ID={ID}")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Sum plot
        total_error_append = np.array(residual_error_append) + np.array(solution_error_append)

        fig3 = plt.figure(figsize=(8, 5))
        plt.plot(iterations, total_error_append, marker='^', linestyle='-', color='tab:green')
        plt.xlabel('Iterations')
        plt.ylabel(r'$F = \frac{1}{2} [\|\mathbf{z}-\mathbf{h}(\mathbf{\theta_0})-\mathbf{J_k}(\mathbf{\theta}-\mathbf{\theta_0})\|_{\mathbf{W_1}}^2 + \beta^2 \|\mathbf{\theta}-\mathbf{\theta^*}\|_{\mathbf{W_2}}^2]$')
        plt.title(r'Functional: $F = \frac{1}{2} [\|\mathbf{z}-\mathbf{h}(\mathbf{\theta_0})-\mathbf{J_k}(\mathbf{\theta}-\mathbf{\theta_0})\|_{\mathbf{W_1}}^2 + \beta^2 \|\mathbf{\theta}-\mathbf{\theta^*}\|_{\mathbf{W_2}}^2]$' + f'   ({vessel})')
        plt.grid(True)
        plt.tight_layout()

        plot_path3 = os.path.join(plots_dir, f"1.Functional")
        plt.savefig(f"{plot_path3}.png", dpi=300)
        plt.savefig(f"{plot_path3}.svg")
        with open(f"{plot_path3}.pkl", "wb") as f:
            pickle.dump(fig3, f)
        plt.close(fig3)

        # 2. residual_error plot
        fig1 = plt.figure(figsize=(8, 5))
        plt.plot(iterations, residual_error_append, marker='o', linestyle='-', color='tab:red')
        plt.xlabel('Iterations')
        plt.ylabel(r'$\frac{1}{2} \|\mathbf{z}-\mathbf{h}(\mathbf{\theta_0})-\mathbf{J_k}(\mathbf{\theta}-\mathbf{\theta_0})\|_{\mathbf{W_1}}^2$')
        plt.title(r'Residual norm: $\frac{1}{2} \|\mathbf{z}-\mathbf{h}(\mathbf{\theta_0})-\mathbf{J_k}(\mathbf{\theta}-\mathbf{\theta_0})\|_{\mathbf{W_1}}^2$' + f'   ({vessel})')
        plt.grid(True)
        plt.tight_layout()

        plot_path1 = os.path.join(plots_dir, f"2.Residual_norm")
        plt.savefig(f"{plot_path1}.png", dpi=300)
        plt.savefig(f"{plot_path1}.svg")
        with open(f"{plot_path1}.pkl", "wb") as f:
            pickle.dump(fig1, f)
        plt.close(fig1)

        # 3. solution_error plot
        fig2 = plt.figure(figsize=(8, 5))
        plt.plot(iterations, solution_error_append, marker='s', linestyle='-', color='tab:blue')
        plt.xlabel('Iterations')
        plt.ylabel(r'$\frac{1}{2} \beta^2 \|\mathbf{\theta}-\mathbf{\theta^*}\|_{\mathbf{W_2}}^2$')
        plt.title(r'Solution norm: $\frac{1}{2}\beta^2 \|\mathbf{\theta}-\mathbf{\theta^*}\|_{\mathbf{W_2}}^2$' + f'   ({vessel})')
        plt.grid(True)
        plt.tight_layout()

        plot_path2 = os.path.join(plots_dir, f"3.Solution_norm")
        plt.savefig(f"{plot_path2}.png", dpi=300)
        plt.savefig(f"{plot_path2}.svg")
        with open(f"{plot_path2}.pkl", "wb") as f:
            pickle.dump(fig2, f)
        plt.close(fig2)

        
        # Confirmation in the terminal
        print("Plots saved:")
        print(f" - {plot_path1}.png, .svg, .pkl")
        print(f" - {plot_path2}.png, .svg, .pkl")
        print(f" - {plot_path3}.png, .svg, .pkl")


    def plot_iter(self, ID, vessel, delta_dict, knumber_max: int):
        """Plots the parameters with delta ≠ 0 and their relative differences from the patient."""

        plt.close('all')

        file_template = 'Pk_{}.last'
        plots_dir = os.path.join(openBF_dir, f"iteration_plots_ID={ID}")
        os.makedirs(plots_dir, exist_ok=True)

        patient_parameters = "Pm"
        patient_yaml = self.patient_file
        self.Pk(vessel, delta_dict, patient_parameters, patient_yaml)

        file_name = file_template.format(vessel)
        folders = ['P0'] + [f'optimized_parameters_P{i}' for i in range(1, knumber_max + 1)]

        all_parameters = ["h0", "L", "R0", "Rp", "Rd", "E", "R1", "R2", "Cc"]
        param_labels = {
            "h0": "Wall thickness (h0) [m]",
            "L": "Length (L) [m]",
            "R0": "Lumen radius (R0) [m]",
            "Rp": "Proximal radius (Rp) [m]",
            "Rd": "Distal radius (Rd) [m]",
            "E": "Elastic modulus (E) [Pa]",
            "R1": "First peripheral resistance (R1) " + r'$[Pa.s.m^{-3}]$',
            "R2": "Second peripheral resistance (R2) " + r'$[Pa.s.m^{-3}]$',
            "Cc": "Peripheral compliance (Cc) " + r'$[m^3.Pa^{-1}]$'       
            }

        valid_params = [p for p in all_parameters if delta_dict.get(p, 0) != 0]
        param_data = {p: [] for p in valid_params}

        for folder in folders:
            file_path = os.path.join(self.openBF_dir, folder, file_name)
            if not os.path.isfile(file_path):
                raise SystemExit(f"Error: File not found at {file_path}. Execution stopped.")

            dados = np.loadtxt(file_path).flatten()
            for i, p in enumerate(valid_params):
                param_data[p].append(dados[i])

        patient_path = os.path.join(self.openBF_dir, patient_parameters, file_name)
        if not os.path.isfile(patient_path):
            raise SystemExit(f"Error: Patient file not found at {patient_path}. Execution stopped.")

        patient_data = np.loadtxt(patient_path).flatten()
        patient_ref = {p: patient_data[i] for i, p in enumerate(valid_params)}

        iterations = np.arange(len(folders))

        # Plots relative differences (all together)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for p in valid_params:
            vals = np.array(param_data[p])
            ref = patient_ref[p]
            diff = np.abs(((vals - ref) / ref) * 100)
            ax2.plot(iterations, diff, marker='o', label=param_labels.get(p, p))
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_title(f'Absolute Relative Difference of Parameters (%)  ({vessel})', fontsize=14)
        ax2.set_xlabel('Iterations', fontsize=14)
        ax2.set_ylabel(r'$\left|\frac{\mathbf{\theta_0} - \mathbf{\theta_z}}{\mathbf{\theta_z}}\right| \times 100 \,\%$', fontsize=14)
        ax2.grid(True)
        ax2.legend()
        fig2.tight_layout()
        rel_diff_path = os.path.join(plots_dir, f"4.Relative_difference")
        fig2.savefig(f"{rel_diff_path}.png", dpi=300)
        fig2.savefig(f"{rel_diff_path}.svg")
        with open(f"{rel_diff_path}.pkl", "wb") as f:
            pickle.dump(fig2, f)
        plt.close(fig2)
        print(f"Saved: {rel_diff_path}.png, .svg, .pkl")

        # Plots absolute values for each valid parameter
        for p in valid_params:
            fig, ax = plt.subplots(figsize=(10, 6))
            y = np.array(param_data[p])
            ref = patient_ref[p]
            label = param_labels.get(p, p)

            # Plots parameter evolution
            ax.plot(iterations, y, 'o-', label = f'Estimated {p} - ' + r'$(\mathbf{\theta_0})$')
            # Patient line
            ax.axhline(ref, linestyle='--', linewidth=2,
                       color=ax.lines[-1].get_color(),
                       label=f'Patient {p} - ' + r'$(\mathbf{\theta_z})$')

            ax.set_title(f'{label} vs Iterations   ({vessel})', fontsize=14)
            ax.set_xlabel('Iterations', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Saves the plot
            plot_path = os.path.join(plots_dir, f"5.Absolute_{p}_values")
            fig.savefig(f"{plot_path}.png", dpi=300)
            fig.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)
            print(f"Saved: {plot_path}.png, .svg, .pkl")


    def file_openBF(self, yaml_file, output_folder_name):
        """Runs openBF in Julia for the specified YAML file
            and stacks the output values."""

        # Where the yaml_file output files will be
        file_dir = os.path.join(openBF_dir, output_folder_name)
        os.makedirs(file_dir, exist_ok=True)

        # Runs openBF to yaml_file
        self.openBF(yaml_file, file_dir)

        # Stack openBF outputs for each vessel individually
        vessels = ["vessel1", "vessel2", "vessel3"]
        files_blocks= [["vessel1_P.last", "vessel1_u.last"],["vessel2_P.last", "vessel2_u.last"],["vessel3_P.last", "vessel3_u.last"]]
        
        for vessel_name, files in zip(vessels, files_blocks):
            self.stack_last_files(files, file_dir, vessel_name)


    def updated_openBF(self, knumber, vessel, parameter, add_value):
        """ Updates the value of a parameter within a specific vessel;
        runs openBF in Julia for the updated YAML;
        stacks the output values;
        calculates the partial derivatives with respect to the modified parameter."""

        # Updates the YAML file to the specified parameter
        self.update_yaml(knumber, vessel, parameter, add_value)

        # Where the k_file output files are
        base_dir = os.path.join(openBF_dir, f"y{knumber}_openBF_output")
        os.makedirs(base_dir, exist_ok=True)
        # Where the updated_file output files will be
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_{vessel}_{parameter}")
        os.makedirs(updated_dir, exist_ok=True)

        # Runs openBF to updated_file
        updated_file = os.path.join(openBF_dir, f"updated_{parameter}.yaml")
        self.file_openBF(updated_file,updated_dir)

        # Path to the partial derivatives directory
        del_dir = os.path.join(openBF_dir, f"partial_deriv_{parameter}")

        # Calculates and creates the partial derivatives files
        self.partial_deriv_files(vessel, base_dir, updated_dir, del_dir, parameter)

    def tied_updated_openBF(self, knumber, parameter, add_value):
        """ Updates the value of a parameter for both vessels 2 and 3;
        runs openBF in Julia for the updated YAML;
        stacks the output values;
        calculates the partial derivatives with respect to the modified parameter."""

        parameters = ["R1", "Cc"]

        # Updates the YAML file to the specified parameter
        for parameter in parameters:
            self.update_yaml_tied(knumber, parameter, add_value)

        # Where the k_file output files are
        base_dir = os.path.join(openBF_dir, f"y{knumber}_openBF_output")
        os.makedirs(base_dir, exist_ok=True)

        # Where the updated_file output files will be
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_WK")
        os.makedirs(updated_dir, exist_ok=True)

        # Runs openBF to updated_file and stacks the openBF output
        updated_file = os.path.join(openBF_dir, f"updated_WK.yaml")
        self.file_openBF(updated_file,updated_dir)

        # Path to the partial derivatives directory
        del_dir = os.path.join(openBF_dir, f"partial_deriv_WK")

        # Calculates and creates the partial derivatives files
        vessels = ["vessel2", "vessel3"]
        parameters = ["R1", "Cc"]
        for vessel in vessels:
            for parameter in parameters:
                self.tied_partial_deriv_files(vessel, base_dir, updated_dir, del_dir, parameter)


    def L_curve(self, vessel, knumber, plot=True):
        # To generate the L-curve and find beta_opt, you don't need a specific β beforehand: 
        # you only need the iterations of the model without regularization or with a fixed 
        # initial β used to generate the files.

        if plot:
            plt.close('all')

        residual_append = []
        solution_append = []
        kplus = knumber + 1

        # Files paths
        patient_file = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_withnoise.last")
        yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"{vessel}_stacked.last")
        Jk_file = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_{vessel}_stacked.txt")
        k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
        kplus_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{kplus}", f"Pk_{vessel}.last")
        if knumber == 0:
            k_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
        else:
            k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_{vessel}.last")
        kstar_param_file = self.kstar_file

        # Checks if files exist
        required_files = [
            patient_file,
            yk_file,
            Jk_file,
            k0_param_file,
            kplus_param_file,
            k_param_file,
            kstar_param_file
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")


        # Loads files ignoring comments
        patient_data = np.loadtxt(patient_file, comments="#")
        yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
        Jk_data = np.loadtxt(Jk_file, comments="#")
        k0_data = np.atleast_1d(np.loadtxt(k0_param_file, comments="#"))
        kplus_data = np.atleast_1d(np.loadtxt(kplus_param_file, comments="#"))
        k_data = np.atleast_1d(np.loadtxt(k_param_file, comments="#"))
        kstar_data = np.atleast_1d(np.loadtxt(kstar_param_file, comments="#"))


        # Corrects Jk dimension 
        if Jk_data.ndim == 1:
            Jk_data = Jk_data.reshape(-1, 1) # (200,) -> (200,1)

        # Creates 1000 points of beta
        beta_values = np.logspace(-8, 2, 1000)   # 1000 points between 1e-8 e 1e2

        for beta in beta_values:

            # Tikhonov Regularization matrices
            z = yk_data
            W1 = np.diag(1 / (z**2)) # Weighting matrix

            P = k0_data
            W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

            # Creates the A matrix
            A = Jk_data.T @ W1 @ Jk_data + beta**2 * W2

            # Creates the B matrix
            R_matrix = patient_data - yk_data
            B = Jk_data.T @ W1 @ R_matrix - beta**2 * W2 @ (k_data - kstar_data)

            # Adjusting the dimensions to use np.lingalg.solve
            if B.ndim == 0:
                B = np.array([B])     # scalar becomes 1D vector
            if B.ndim == 1:
                B = B.reshape(-1, 1)  # vector becomes column

            # Solves the equation with the corresponding beta
            try:
                dp = np.linalg.solve(A, B)
                dp = dp.ravel()   # Ensures vector (n_params,)
            except np.linalg.LinAlgError:
                raise SystemExit(f"Error: A is not singular (non inversible). Execution stopped.")
            

            # Calculates the squared error of the output of iteration k with respect to the patient output
            residual = patient_data - yk_data - Jk_data @ dp
            residual_norm = (residual.T @ W1 @ residual)

            # Stores it to plot
            residual_append.append(residual_norm)

            # Calculates the squared error of the parameters
            solution = kplus_data - kstar_data # Pk+1 - P*
            solution_norm = (solution.T @ W2 @ solution)

            # Stores it to plot
            solution_append.append(solution_norm)

        # Low-pass filter
        low_residual = savgol_filter(residual_append, window_length=15, polyorder=3)
        low_solution = savgol_filter(solution_append, window_length=15, polyorder=3)


        # x_log = np.log(np.array(low_residual))
        # y_log = np.log(np.array(low_solution))

        # # Finding the point of minimum distance of the origin
        # idx_opt = np.argmin(np.sqrt(x_log**2 + y_log**2))

        x = np.log(np.array(low_residual))
        y = np.log(np.array(low_solution))

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature)  # replaces NaN for 0

        idx_opt = np.argmax(curvature)


        beta_opt = beta_values[idx_opt]
        print(f"Optimal β found: {beta_opt:.3e}")
    
        if plot:
            # Plot directory
            plot_dir = os.path.join(openBF_dir, "iteration_plots_Lcurve")
            os.makedirs(plot_dir, exist_ok=True)

            # Plot
            fig = plt.figure(figsize=(11, 6))
            plt.loglog(low_residual, low_solution, marker='o', linestyle='-', color='tab:red')
            plt.scatter(low_residual[idx_opt], low_solution[idx_opt], color='blue', s=80, label=f"β ótimo={beta_opt:.1e}")
            plt.xlabel('Residual norm')
            plt.ylabel('Solution norm')
            plt.title(f'L-curve - {vessel}')
            plt.legend()
            plt.grid(True, which="both")


            plot_path = os.path.join(plot_dir, f"Lcurve_{vessel}_k{knumber}")
            plt.savefig(f"{plot_path}.png", dpi=300)
            plt.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)

            # Confirmaton on the terminal
            print(f"L-curve plot saved: {plot_path}.png, .svg, .pkl")

        # Creates a dictionary with the beta's
        if not hasattr(self, "Lcurve_dict"):
            self.Lcurve_dict = {}
        
        self.Lcurve_dict[knumber] = beta_opt

        # Returns the beta corresponding to iteration k
        return beta_opt
    

    def tied_L_curve(self, knumber, plot=True):
        # To generate the L-curve and find beta_opt, you don't need a specific β beforehand: 
        # you only need the iterations of the model without regularization or with a fixed 
        # initial β used to generate the files.

        if plot:
            plt.close('all')

        residual_append = []
        solution_append = []
        kplus = knumber + 1

        # Files paths
        patient_file = os.path.join(openBF_dir, "ym_openBF_patient_output", f"vessels2and3_withnoise_stacked.last")
        yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"vessels2and3_stacked.last")
        Jk_file = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_vessels2and3.txt")
        k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_vessel2.last")
        kplus_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{kplus}", f"Pk_vessels2and3.last")
        if knumber == 0:
            k_param_file = os.path.join(openBF_dir, "P0", f"Pk_vessel2.last")
        else:
            k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_vessels2and3.last")
        kstar_param_file = self.kstar_file

        # Checks if files exist
        required_files = [
            patient_file,
            yk_file,
            Jk_file,
            k0_param_file,
            kplus_param_file,
            k_param_file,
            kstar_param_file
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")


        # Loads files ignoring comments
        patient_data = np.loadtxt(patient_file, comments="#")
        yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
        Jk_data = np.loadtxt(Jk_file, comments="#")
        k0_data = np.atleast_1d(np.loadtxt(k0_param_file, comments="#"))
        kplus_data = np.atleast_1d(np.loadtxt(kplus_param_file, comments="#"))
        k_data = np.atleast_1d(np.loadtxt(k_param_file, comments="#"))
        kstar_data = np.atleast_1d(np.loadtxt(kstar_param_file, comments="#"))


        # Corrects Jk dimension 
        if Jk_data.ndim == 1:
            Jk_data = Jk_data.reshape(-1, 1) # (200,) -> (200,1)

        # Creates 1000 points of beta
        beta_values = np.logspace(-8, 2, 1000)   # 1000 points between 1e-8 e 1e2

        for beta in beta_values:

            # Tikhonov Regularization matrices
            z = yk_data
            W1 = np.diag(1 / (z**2)) # Weighting matrix

            P = k0_data
            W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

            # Creates the A matrix
            A = Jk_data.T @ W1 @ Jk_data + beta**2 * W2

            # Creates the B matrix
            R_matrix = patient_data - yk_data
            B = Jk_data.T @ W1 @ R_matrix - beta**2 * W2 @ (k_data - kstar_data)

            # Adjusting the dimensions to use np.lingalg.solve
            if B.ndim == 0:
                B = np.array([B])     # scalar becomes 1D vector
            if B.ndim == 1:
                B = B.reshape(-1, 1)  # vector becomes column

            # Solves the equation with the corresponding beta
            try:
                dp = np.linalg.solve(A, B)
                dp = dp.ravel()   # Ensures vector (n_params,)
            except np.linalg.LinAlgError:
                raise SystemExit(f"Error: A is not singular (non inversible). Execution stopped.")
            

            # Calculates the squared error of the output of iteration k with respect to the patient output
            residual = patient_data - yk_data - Jk_data @ dp
            residual_norm = (residual.T @ W1 @ residual)

            # Stores it to plot
            residual_append.append(residual_norm)

            # Calculates the squared error of the parameters
            solution = kplus_data - kstar_data # Pk+1 - P*
            solution_norm = (solution.T @ W2 @ solution)

            # Stores it to plot
            solution_append.append(solution_norm)

        # Low-pass filter
        low_residual = savgol_filter(residual_append, window_length=15, polyorder=3)
        low_solution = savgol_filter(solution_append, window_length=15, polyorder=3)


        # x_log = np.log(np.array(low_residual))
        # y_log = np.log(np.array(low_solution))

        # # Finding the point of minimum distance of the origin
        # idx_opt = np.argmin(np.sqrt(x_log**2 + y_log**2))

        x = np.log(np.array(low_residual))
        y = np.log(np.array(low_solution))

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature)  # replaces NaN for 0

        idx_opt = np.argmax(curvature)


        beta_opt = beta_values[idx_opt]
        print(f"Optimal β found: {beta_opt:.3e}")
    
        if plot:
            # Plot directory
            plot_dir = os.path.join(openBF_dir, "iteration_plots_Lcurve")
            os.makedirs(plot_dir, exist_ok=True)

            # Plot
            fig = plt.figure(figsize=(11, 6))
            plt.loglog(low_residual, low_solution, marker='o', linestyle='-', color='tab:red')
            plt.scatter(low_residual[idx_opt], low_solution[idx_opt], color='blue', s=80, label=f"β ótimo={beta_opt:.1e}")
            plt.xlabel('Residual norm')
            plt.ylabel('Solution norm')
            plt.title(f'L-curve - Vessels 2 and 3 tied')
            plt.legend()
            plt.grid(True, which="both")


            plot_path = os.path.join(plot_dir, f"Lcurve_WK_k{knumber}")
            plt.savefig(f"{plot_path}.png", dpi=300)
            plt.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)

            # Confirmaton on the terminal
            print(f"L-curve plot saved: {plot_path}.png, .svg, .pkl")

        # Creates a dictionary with the beta's
        if not hasattr(self, "Lcurve_dict"):
            self.Lcurve_dict = {}
        
        self.Lcurve_dict[knumber] = beta_opt

        # Returns the beta corresponding to iteration k
        return beta_opt
   

    def Morozov(self, vessel, knumber, plot=True):
    
        if plot:
            plt.close('all')

        discrepancy = []

        # Files paths
        patient_file = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_withnoise.last")
        patient_file_withoutnoise = os.path.join(openBF_dir, "ym_openBF_patient_output", f"{vessel}_stacked.last")
        yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"{vessel}_stacked.last")
        Jk_file = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_{vessel}_stacked.txt")
        k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
        if knumber == 0:
            k_param_file = os.path.join(openBF_dir, "P0", f"Pk_{vessel}.last")
        else:
            k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_{vessel}.last")
        kstar_param_file = self.kstar_file

        # Checks if files exist
        required_files = [
            patient_file,
            patient_file_withoutnoise,
            yk_file,
            Jk_file,
            k0_param_file,
            k_param_file,
            kstar_param_file
        ]


        for file_path in required_files:
            if not os.path.exists(file_path):
                raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")


        # Loads files ignoring comments
        patient_data = np.loadtxt(patient_file, comments="#")
        patient_data_withoutnoise = np.loadtxt(patient_file_withoutnoise, comments="#")[:, 3] # Takes only the 4ª column
        yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
        Jk_data = np.loadtxt(Jk_file, comments="#")
        k0_data = np.atleast_1d(np.loadtxt(k0_param_file, comments="#"))
        k_data = np.atleast_1d(np.loadtxt(k_param_file, comments="#"))
        kstar_data = np.atleast_1d(np.loadtxt(kstar_param_file, comments="#"))


        # Corrects Jk dimension 
        if Jk_data.ndim == 1:
            Jk_data = Jk_data.reshape(-1, 1) # (200,) -> (200,1)

        # Creates 1000 points of beta
        beta_values = np.logspace(-8, 2, 1000)   # 1000 points between 1e-8 e 1e2

        # Creates the sigma vector
        m = len(yk_data)
        sigma = np.zeros(m)

        # First 100 rows with pressure standard deviation
        pressure_std = 266.64 # [Pa]
        sigma[:100] = 1/pressure_std

        # Last 100 rows with velocity standard deviation
        velocity_mean = np.mean(patient_data_withoutnoise[-100:])  
        velocity_std = 0.03 * velocity_mean # [m/s]
        sigma[-100:] = 1/velocity_std

        # Creates a diagonal matrix with sigma
        sigma_diag = np.diag(sigma)   # (200, 200)

        # Defines delta
        n = len(k0_data) 
        delta = m - n
        print("Delta = ", delta)

        # Tikhonov Regularization matrices
        z = yk_data
        W1 = np.diag(1 / (z**2)) # Weighting matrix

        P = k0_data
        W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

        for beta in beta_values:

            # Creates the A matrix
            A = Jk_data.T @ W1 @ Jk_data + beta**2 * W2

            # Creates the B matrix
            R_matrix = patient_data - yk_data
            B = Jk_data.T @ W1 @ R_matrix - beta**2 * W2 @ (k_data - kstar_data)

            # Adjusting the dimensions to use np.lingalg.solve
            if B.ndim == 0:
                B = np.array([B])     # scalar becomes 1D vector
            if B.ndim == 1:
                B = B.reshape(-1, 1)  # vector becomes column

            # Solves the equation with the corresponding beta
            try:
                dp = np.linalg.solve(A, B)
                dp = dp.ravel()   # Ensures vector (n_params,)
            except np.linalg.LinAlgError:
                raise SystemExit(f"Error: A is not singular (non-invertible). Execution stopped.")
            
            # Calculates the residual norm
            residual = patient_data - yk_data - Jk_data @ dp
            weighted_residual = sigma_diag @ residual
            residual_norm = np.linalg.norm(weighted_residual)

            # Difference from expected noise level
            discrepancy.append(abs(residual_norm - delta))

        # Finds the optimal beta index
        idx_opt = int(np.argmin(np.abs(discrepancy)))
        beta_opt = beta_values[idx_opt]

        if plot:

            # Plot directory
            plot_dir = os.path.join(openBF_dir, "iteration_plots_discrepancy")
            os.makedirs(plot_dir, exist_ok=True)

            fig = plt.figure(figsize=(11, 6))
            plt.loglog(beta_values, discrepancy)
            plt.axvline(beta_opt, color='red', linestyle='--')
            plt.xlabel('β')
            plt.ylabel('Discrepancy: ' + r'$ [\| \mathbf{\sigma} \times [\mathbf{z}-\mathbf{h}(\mathbf{\theta_0})-\mathbf{J_k}(\mathbf{\theta}-\mathbf{\theta_0})]\|_2 - \mathbf{\delta}]$', fontsize = 14)
            plt.title("Morozov's Discrepancy Principle")
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(plot_dir, f"discrepancy_{vessel}_k{knumber}")
            plt.savefig(f"{plot_path}.png", dpi=300)
            plt.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)

            # Confirmaton on the terminal
            print(f"Discrepancy Principle plot saved: {plot_path}.png, .svg, .pkl")


        # Creates a dictionary with the beta's
        if not hasattr(self, "Morozov_dict"):
            self.Morozov_dict = {}
        
        self.Morozov_dict[knumber] = beta_opt

        # Returns the beta corresponding to iteration k
        print(f"The optimal beta for k={knumber} is:", beta_opt)
        return beta_opt


    def tied_Morozov(self, knumber, plot=True):
    
        if plot:
            plt.close('all')

        discrepancy = []

        # Files paths
        patient_file = os.path.join(openBF_dir, "ym_openBF_patient_output", f"vessels2and3_withnoise_stacked.last")
        patient_file_withoutnoise = os.path.join(openBF_dir, "ym_openBF_patient_output", f"vessels2and3_stacked.last")
        yk_file = os.path.join(openBF_dir, f"y{knumber}_openBF_output", f"vessels2and3_stacked.last")
        Jk_file = os.path.join(openBF_dir, f"jacobians", f"jacobian_k={knumber}_vessels2and3.txt")
        k0_param_file = os.path.join(openBF_dir, "P0", f"Pk_vessel2.last")
        if knumber == 0:
            k_param_file = k0_param_file
        else:
            k_param_file = os.path.join(openBF_dir, f"optimized_parameters_P{knumber}", f"Pk_vessels2and3.last")
        kstar_param_file = self.kstar_file

        # Checks if files exist
        required_files = [
            patient_file,
            patient_file_withoutnoise,
            yk_file,
            Jk_file,
            k0_param_file,
            k_param_file,
            kstar_param_file
        ]


        for file_path in required_files:
            if not os.path.exists(file_path):
                raise SystemExit(f"Error: Required file '{file_path}' not found. Execution stopped.")


        # Loads files ignoring comments
        patient_data = np.loadtxt(patient_file, comments="#")
        patient_data_withoutnoise = np.loadtxt(patient_file_withoutnoise, comments="#")[:, 3] # Takes only the 4ª column
        yk_data = np.loadtxt(yk_file, comments="#")[:, 3] # Takes only the 4ª column
        Jk_data = np.loadtxt(Jk_file, comments="#")
        k0_data = np.atleast_1d(np.loadtxt(k0_param_file, comments="#"))
        k_data = np.atleast_1d(np.loadtxt(k_param_file, comments="#"))
        kstar_data = np.atleast_1d(np.loadtxt(kstar_param_file, comments="#"))


        # Corrects Jk dimension 
        if Jk_data.ndim == 1:
            Jk_data = Jk_data.reshape(-1, 1) # (200,) -> (200,1)

        # Creates 1000 points of beta
        beta_values = np.logspace(-8, 2, 1000)   # 1000 points between 1e-8 e 1e2

        # Creates the sigma vector
        m = len(yk_data)
        sigma = np.zeros(m)

        # First 100 rows with pressure standard deviation
        pressure_std = 266.64 # [Pa]
        sigma[:100] = 1/pressure_std

        # Last 100 rows with velocity standard deviation
        velocity_mean = np.mean(patient_data_withoutnoise[-100:])  
        velocity_std = 0.03 * velocity_mean # [m/s]
        sigma[-100:] = 1/velocity_std

        # Creates a diagonal matrix with sigma
        sigma_diag = np.diag(sigma)   # (200, 200)

        # Defines delta
        n = len(k0_data) 
        delta = m - n
        print("Delta = ", delta)

        # Tikhonov Regularization matrices
        z = yk_data
        W1 = np.diag(1 / (z**2)) # Weighting matrix

        P = k0_data
        W2 = np.diag(1 / (P**2)) # W2=L2.T@L2, L2 = Regularization matrix

        for beta in beta_values:

            # Creates the A matrix
            A = Jk_data.T @ W1 @ Jk_data + beta**2 * W2

            # Creates the B matrix
            R_matrix = patient_data - yk_data
            B = Jk_data.T @ W1 @ R_matrix - beta**2 * W2 @ (k_data - kstar_data)

            # Adjusting the dimensions to use np.lingalg.solve
            if B.ndim == 0:
                B = np.array([B])     # scalar becomes 1D vector
            if B.ndim == 1:
                B = B.reshape(-1, 1)  # vector becomes column

            # Solves the equation with the corresponding beta
            try:
                dp = np.linalg.solve(A, B)
                dp = dp.ravel()   # Ensures vector (n_params,)
            except np.linalg.LinAlgError:
                raise SystemExit(f"Error: A is not singular (non-invertible). Execution stopped.")
            
            # Calculates the residual norm
            residual = patient_data - yk_data - Jk_data @ dp
            weighted_residual = sigma_diag @ residual
            residual_norm = np.linalg.norm(weighted_residual)

            # Difference from expected noise level
            discrepancy.append(abs(residual_norm - delta))

        # Finds the optimal beta index
        idx_opt = int(np.argmin(np.abs(discrepancy)))
        beta_opt = beta_values[idx_opt]

        if plot:

            # Plot directory
            plot_dir = os.path.join(openBF_dir, "iteration_plots_discrepancy")
            os.makedirs(plot_dir, exist_ok=True)

            fig = plt.figure(figsize=(11, 6))
            plt.loglog(beta_values, discrepancy)
            plt.axvline(beta_opt, color='red', linestyle='--')
            plt.xlabel('β')
            plt.ylabel('Discrepancy: ' + r'$ [\| \mathbf{\sigma} \times [\mathbf{z}-\mathbf{h}(\mathbf{\theta_0})-\mathbf{J_k}(\mathbf{\theta}-\mathbf{\theta_0})]\|_2 - \mathbf{\delta}]$', fontsize = 14)
            plt.title("Morozov's Discrepancy Principle")
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(plot_dir, f"discrepancy_WK_k{knumber}")
            plt.savefig(f"{plot_path}.png", dpi=300)
            plt.savefig(f"{plot_path}.svg")
            with open(f"{plot_path}.pkl", "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)

            # Confirmaton on the terminal
            print(f"Discrepancy Principle plot saved: {plot_path}.png, .svg, .pkl")


        # Creates a dictionary with the beta's
        if not hasattr(self, "Morozov_dict"):
            self.Morozov_dict = {}
        
        self.Morozov_dict[knumber] = beta_opt

        # Returns the beta corresponding to iteration k
        print(f"The optimal beta for k={knumber} is:", beta_opt)
        return beta_opt


    def iteration(self, ID, knumber, vessel, beta_method, alpha, delta_dict):
        """Runs an iteration to obtain the optimized parameters P(k+1)."""

        print(f"\n=== Starting iteration {knumber}, ID: {ID} ===\n")

        # Filters parameters with delta != 0
        valid_parameters = [param for param in delta_dict if delta_dict[param] != 0]
        print (f"The valid parameters are: {valid_parameters}.")

        if knumber == 0:
            k_yaml_file = os.path.join(openBF_dir, self.k0_file)

            # Checks if file exists
            if not os.path.exists(k_yaml_file):
                raise SystemExit(f"Error: File {k_yaml_file} not found. Execution stopped.")

            # Runs openBF to 0-iteration YAML file
            self.file_openBF(k_yaml_file, f"y{knumber}_openBF_output")

        # Updates the paramater and calculates its partial derivative
        for parameter in valid_parameters:
            self.updated_openBF(knumber, vessel, parameter, delta_dict[parameter])

        # Path to the Jacobian matrices directory
        output_path = os.path.join(openBF_dir, f"jacobians")
        self.stack_partial_derivatives(knumber, vessel, delta_dict, output_path)

        # Creates the P0 matrix (parameters of the k-iteration yaml)
        if knumber == 0:
            yaml_file = self.k0_file
            param_directory = "P0"

            self.Pk(vessel, delta_dict, param_directory, yaml_file)


        if beta_method == "L_curve":
            # Calculates the optimized parameters with a initial guess for beta
            beta_guess = 1e-4  
            self.A_matrix(ID, beta_guess, vessel, knumber)
            self.B_matrix(ID, beta_guess, vessel, knumber)
            self.optimized_parameters(ID, vessel, alpha, beta_guess, knumber)

            # Finds optimal beta
            beta_opt = self.L_curve(vessel, knumber, plot=True)

        if beta_method == "Morozov":
            # Finds optimal beta
            beta_opt = self.Morozov(vessel, knumber, plot=True)

        else:
            raise ValueError(f"Unknown beta_method: {beta_method}. Please insert 'L_curve' or 'Morozov'.")

        # Creates the A matrix
        self.A_matrix(ID, beta_opt, vessel, knumber)

        # Creates the B matrix
        self.B_matrix(ID, beta_opt, vessel, knumber)

        # Creates the optimized parameters matrix
        self.optimized_parameters(ID, vessel, alpha, beta_opt, knumber)

        # Updates YAML with optimized parameters
        if knumber == 0:
            base_yaml_path = os.path.join(openBF_dir, self.k0_file)
        else:
            base_yaml_path = os.path.join(openBF_dir, f"inverse_problem_k={knumber}.yaml")
        opt_param_files_dir = os.path.join(openBF_dir, f"optimized_parameters_P{knumber+1}")
        opt_output_yaml_path = os.path.join(openBF_dir, f"inverse_problem_k={knumber+1}.yaml")

        # Checks if file exists
        if not os.path.exists(base_yaml_path):
            raise SystemExit(f"Error: File {base_yaml_path} not found. Execution stopped.")

        self.update_yaml_with_optimized_parameters(vessel, delta_dict, base_yaml_path, opt_param_files_dir, opt_output_yaml_path)

        # Runs openBF to the new/optimized yaml file
        self.file_openBF(opt_output_yaml_path, f"y{knumber+1}_openBF_output")

    
    def tied_iteration(self, ID, knumber, beta_method, alpha, delta_dict):
        """Runs an iteration to obtain the optimized parameters P(k+1) for R1 and Cc. Updates vessels 2 and 3 simultaneously with the same value."""

        print(f"\n=== Starting iteration {knumber}, ID: {ID} ===\n")

        # Defines valid parameters
        # Warning: Considering a two-element Windkessel model
        valid_parameters = ["R1", "Cc"]
        print (f"The valid parameters are: {valid_parameters}.")

        # Runs simulation for k=0
        if knumber == 0:
            k_yaml_file = os.path.join(openBF_dir, self.k0_file)

            # Checks if file exists
            if not os.path.exists(k_yaml_file):
                raise SystemExit(f"Error: File {k_yaml_file} not found. Execution stopped.")
            
            # Checks if the initial guess for R1 and Cc in vessels 2 and 3 are equal
            if self.equal_parameters(k_yaml_file)==True:
                # Runs openBF to 0-iteration YAML files
                self.file_openBF(k_yaml_file, f"y{knumber}_openBF_output")
            else: 
                raise SystemExit(f"Error: Please, insert the same values for R1 and Cc in vessels 2 and 3: {k_yaml_file}.")

        # Updates R1 and Cc in vessels 2 and 3 and calculates their partial derivatives
        for parameter in valid_parameters:
            self.tied_updated_openBF(knumber, parameter, delta_dict[parameter])

        # Creates the Jacobian matrix
        output_path = os.path.join(openBF_dir, f"jacobians")
        self.tied_stack_partial_derivatives(knumber, delta_dict, output_path)

        # Creates the P0 matrix (parameters of the k-iteration yaml)
        if knumber == 0:
            yaml_file = self.k0_file
            param_directory = "P0"

            # Gets the parameters for vessels 2 and 3 at iteration 0 (they must be the same)
            vessels = ["vessel2", "vessel3"]
            for vessel_P0 in vessels:
                self.Pk(vessel_P0, delta_dict, param_directory, yaml_file)

        # Stacks patient openBF output from vessels 2 and 3 (without noise)
        files= [["vessel2_stacked.last", "vessel3_stacked.last"]]
        file_dir = os.path.join(openBF_dir, "ym_openBF_patient_output")
        
        self.stack_last_files(files, file_dir, "vessels2and3")

        # Stacks patient openBF output from vessels 2 and 3 (with noise)
        files= [["vessel2_withnoise.last", "vessel3_withnoise.last"]]
        file_dir = os.path.join(openBF_dir, "ym_openBF_patient_output")
        
        self.stack_last_files(files, file_dir, "vessels2and3_withnoise")

        # Stacks simulation output from vessels 2 and 3
        files= [["vessel2_stacked.last", "vessel3_stacked.last"]]
        file_dir = os.path.join(openBF_dir, "y{knumber}_openBF_output")
        
        self.stack_last_files(files, file_dir, "vessels2and3")


        if beta_method == "L_curve":
            # Calculates the optimized parameters with a initial guess for beta
            beta_guess = 1e-4  
            self.tied_A_matrix(ID, beta_guess, knumber)
            self.tied_B_matrix(ID, beta_guess, knumber) 
            self.tied_optimized_parameters(ID, alpha, beta_guess, knumber) 

            # Finds optimal beta
            beta_opt = self.tied_L_curve(knumber, plot=True)

        if beta_method == "Morozov":
            # Finds optimal beta
            beta_opt = self.tied_Morozov(knumber, plot=True) # Voltar daqui

        else:
            raise ValueError(f"Unknown beta_method: {beta_method}. Please insert 'L_curve' or 'Morozov'.")

        # Creates the A matrix
        self.tied_A_matrix(ID, beta_opt, knumber)

        # Creates the B matrix
        self.tied_B_matrix(ID, beta_opt, knumber)

        # Creates the optimized parameters matrix
        self.tied_optimized_parameters(ID, alpha, beta_opt, knumber)

        # Updates YAML with optimized parameters
        if knumber == 0:
            base_yaml_path = os.path.join(openBF_dir, self.k0_file)
        else:
            base_yaml_path = os.path.join(openBF_dir, f"inverse_problem_k={knumber}.yaml")
        opt_param_files_dir = os.path.join(openBF_dir, f"optimized_parameters_P{knumber+1}")
        opt_output_yaml_path = os.path.join(openBF_dir, f"inverse_problem_k={knumber+1}.yaml")

        # Checks if file exists
        if not os.path.exists(base_yaml_path):
            raise SystemExit(f"Error: File {base_yaml_path} not found. Execution stopped.")

        self.tied_update_yaml_with_optimized_parameters(delta_dict, base_yaml_path, opt_param_files_dir, opt_output_yaml_path)

        # Runs openBF to the new/optimized yaml file
        self.file_openBF(opt_output_yaml_path, f"y{knumber+1}_openBF_output")


    def search_opt(self, ID, vessel, beta_method, alpha, delta_dict, knumber_max, tied_iteration=False):

        # For the user's knowledge
        if tied_iteration and vessel != "NA":
            raise SystemExit(f"'vessel' is ignored when tied_iteration=True. Use 'NA' for clarity.")

        # Filters parameters with delta != 0
        parameters = ["h0", "L", "R0", "Rp", "Rd", "E", "R1", "R2", "Cc"]
        valid_parameters = [param for param in parameters if delta_dict[param] != 0]

        # Starts chronometer
        start = time.time()

        if tied_iteration==True:

            # Checks if only R1 and Cc are valid parameters
            if valid_parameters == {"R1","Cc"}:

                vessels = ["vessel2", "vessel3"]

                # Adds noise to patient output files
                for vessel_noise in vessels: 
                    self.add_noise(vessel_noise)

                # Begins iteration with R1 and Cc of vessels 2 and 3 tied
                for knumber in range(0, knumber_max + 1):
                    self.tied_iteration(ID, knumber, beta_method, alpha, delta_dict)

                for vessel_plot in vessels:
                    # Plots error for k from 0 to knumber_max
                    self.plot_error(ID, vessel_plot, beta_method, knumber_max)

                    # Plots the parameters evolution 
                    self.plot_iter(ID, vessel_plot, delta_dict, knumber_max)

            else:
                raise SystemExit(f"Error: Tied iteration requires that the only valid parameters be R1 and Cc. Execution stopped.")
            
        else:
            # Adds noise to patient output file 
            self.add_noise(vessel)

            # Runs iteration for k from 0 to knumber_max
            for knumber in range(0, knumber_max + 1):
                self.iteration(ID, knumber, vessel, beta_method, alpha, delta_dict)

            # Plots error for k from 0 to knumber_max
            self.plot_error(ID, vessel, beta_method, knumber_max)

            # Plots the parameters evolution
            self.plot_iter(ID, vessel, delta_dict, knumber_max)

        # Ends chronometer and prints time
        end = time.time()
        minutes = (end - start)/60
        print(f"Elapsed time: {minutes:.3f} minutes.")



# Application
if __name__ == "__main__":

    openBF_dir = "C:/Users/Reinaldo/Documents/inverse_problem_results_vessel2"
    inlet_dat = "C:/Users/Reinaldo/Documents/inverse_problem_results_vessel2/circle_of_willis_inlet.dat"
    patient_yaml = "C:/Users/Reinaldo/Documents/inverse_problem_results_vessel2/inverse_problem_Patient.yaml"
    k0_yaml = "C:/Users/Reinaldo/Documents/inverse_problem_results_vessel2/inverse_problem_k=0_fixed_vessels_1and3.yaml"
    kstar_txt = "C:/Users/Reinaldo/Documents/inverse_problem_results_vessel2/P_star_vessel2.txt"

    updater = OPENBF_Jacobian(openBF_dir, inlet_dat, patient_yaml, k0_yaml, kstar_txt)

    # Runs openBF to patient file
    #updater.file_openBF(patient_yaml, "ym_openBF_patient_output")

    # Searches optimized parameters
    add_values = {"h0": 0.00001, "L": 0.01, "R0": 0.0001, "Rp": 0, "Rd": 0, "E": 0, "R1": 0, "R2": 0, "Cc": 0}
    updater.search_opt(23, "vessel2", "Morozov", 0.3, add_values, 20, tied_iteration=True)
    #updater.search_opt(21, "vessel3", "Morozov", 0.3, 0, 0, 0, 0, 0, 0, 1e6, 0, 1e-13, 20)



