import os
import numpy as np

class test_Jacobian:

    def __init__(self, files_dir):
        self.files_dir = files_dir

    def function(self, xyz_values, delta):
        # Garante que o diretório existe
        base_dir = self.files_dir
        os.makedirs(base_dir, exist_ok=True)

        # Pega os valores de xyz
        xyz_data = np.loadtxt(xyz_values)
        base_x = xyz_data[0]
        base_y = xyz_data[1]
        base_z = xyz_data[2]

        # Armazena os valores da função base
        base_function = np.array([
            base_x * base_y * base_z,
            base_x ** 2 * np.sqrt(base_y) * base_z ** 3,
            base_x * base_y ** 2 * base_z
        ])

        # Salva o arquivo base
        base_path = os.path.join(base_dir, "base_file.txt")
        np.savetxt(base_path, base_function, fmt="%.14e")
        print(f"Base file saved: {base_path}")

        # Percorre todos os parâmetros
        parameters = ['x', 'y', 'z']

        matrices = []

        for param in parameters:
            # Garante que o diretório do arquivo atualizado exista
            updated_value_path = os.path.join(base_dir, f"updated_{param}_value.txt")

            # Atualiza os valores de xyz
            xyz_data = np.loadtxt(xyz_values)
            updated_x = xyz_data[0]
            updated_y = xyz_data[1]
            updated_z = xyz_data[2]

            # Atualiza o valor correspondente
            if param == 'x':
                updated_x += delta
            elif param == 'y':
                updated_y += delta
            elif param == 'z':
                updated_z += delta

            # Salva os valores atualizados
            updated_xyz = np.array([updated_x, updated_y, updated_z])
            np.savetxt(updated_value_path, updated_xyz, fmt="%.14e")
            print(f"Updated {param} value saved: {updated_value_path}")

            # Armazena os valores da função
            updated_function = np.array([
                updated_x * updated_y * updated_z,
                updated_x ** 2 * np.sqrt(updated_y) * updated_z ** 3,
                updated_x * updated_y ** 2 * updated_z
            ])

            # Salva a função atualizada
            updated_path = os.path.join(base_dir, f"updated_function_{param}.txt")
            np.savetxt(updated_path, updated_function, fmt="%.14e")
            print(f"Updated file saved: {updated_path}")

            partial_deriv = (updated_function - base_function)/delta

            # Salva a derivada parcial
            partial_deriv_path = os.path.join(base_dir, f"partial_deriv_{param}.txt")
            np.savetxt(partial_deriv_path, partial_deriv, fmt="%.14e")
            print(f"Updated file saved: {partial_deriv_path}")

            # Empilho as derivadas parciais
            matrices.append(partial_deriv)

            stacked_matrix = np.column_stack(matrices)
            output_file = os.path.join(base_dir, f"jacobian.txt")
            np.savetxt(output_file, stacked_matrix, fmt="%.14e")
            print(f"Stacked matrix saved in: {output_file}")

# Application
if __name__ == "__main__":

    files_dir = "C:/Users/Reinaldo/Documents/jacobian_test"
    teste = test_Jacobian(files_dir)

    xyz_values = "C:/Users/Reinaldo/Documents/jacobian_test/xyz_values.txt"
    delta = 0.0001
    teste.function(xyz_values, delta)
