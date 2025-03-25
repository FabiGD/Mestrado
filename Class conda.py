import yaml
import julia
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

from julia import Main

class OPENBF_Jacobean:
    def __init__(self, base_file, output_file, openBF_dir):
        self.base_file = base_file
        self.output_file = output_file
        self.openBF_dir = openBF_dir

    def load_yaml(self):
        """Carrega o YAML do base_file"""
        with open(self.base_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save_yaml(self, data):
        """Salva o YAML atualizado no output_file"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ Arquivo atualizado salvo em: {self.output_file}")

    def update_yaml(self, vaso, parametro, novo_valor):
        """Atualiza o valor de um parâmetro dentro do vaso especificado"""
        yaml_data = self.load_yaml()

        # Depuração: mostrar estrutura antes da modificação
        print("🔍 Estrutura YAML carregada:", yaml_data)

        if "network" not in yaml_data:
            print("❌ A chave 'network' não foi encontrada no YAML.")
            return

        encontrado = False  # Flag para saber se encontrou o vaso

        for item in yaml_data["network"]:
            print(f"🔍 Verificando vaso: {item.get('label')}")  # Debug print
            if item.get("label") == vaso:
                if parametro in item:
                    item[parametro] = novo_valor
                    print(f"✅ Parâmetro '{parametro}' do vaso '{vaso}' atualizado para: {novo_valor}")
                    encontrado = True
                else:
                    print(f"⚠️ Parâmetro '{parametro}' não encontrado no vaso '{vaso}'.")
                break

        if not encontrado:
            print(f"❌ Vaso '{vaso}' não encontrado no YAML.")

        self.save_yaml(yaml_data)

        # Creates folder for saving plots
        base_dir = os.path.join(openBF_dir, "openBF_base") # Onde estarão os arquivos de saída do base_file
        updated_dir = os.path.join(openBF_dir, f"openBF_updated_{vaso}_{parametro}={novo_valor}") # Onde estarão os arquivos de saída do output_file
        os.makedirs(base_dir, exist_ok=True)
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

        # Runs hemodynamic simulation for base file
        Main.eval(""" 
                    using openBF
                    function pyopenBF(base_file, base_dir)
                        run_simulation(base_file, verbose=true, save_stats=true, savedir=base_dir)
                        println("openBF output saved in: $base_dir")
                    end
                """)

        # Calls Julia function from Python
        Main.pyopenBF(base_file, base_dir)

        # Runs hemodynamic simulation for output file
        Main.eval(""" 
            using openBF
            function pyopenBF(output_file, updated_dir)
                run_simulation(output_file, verbose=true, save_stats=true, savedir=updated_dir)
                println("openBF output saved in: $updated_dir")
            end
        """)

        # Calls Julia function from Python
        Main.pyopenBF(output_file, updated_dir)

        #def jacobian()
        # Vasos e variáveis na ordem correta
        vessels = ["vaso1", "vaso2", "vaso3"]
        variables = ["A", "P", "Q", "u"]  # Ordem correta

        # Função para empilhar os dados corretamente
        def stack_last_files(vessel, variables, data_dir):
            data_list = []

            for var in variables:
                file_path = os.path.join(data_dir, f"{vessel}_{var}.last")

                # Verifica se o arquivo existe
                if not os.path.exists(file_path):
                    print(f"Aviso: Arquivo {file_path} não encontrado.")
                    return

                # Lê os dados garantindo que a estrutura não seja alterada
                data = np.loadtxt(file_path)  # Lê o arquivo mantendo os valores originais
                data_list.append(data)

            # Empilha verticalmente os arquivos (A, P, Q, u)
            stacked_data = np.vstack(data_list)

            # Salva o arquivo empilhado
            output_file = os.path.join(data_dir, f"{vessel}_stacked.last")
            np.savetxt(output_file, stacked_data, fmt="%.6e")  # Mantém formato numérico original

            print(f"Arquivo salvo: {output_file}")

        # Processa cada vaso individualmente
        for vessel in vessels:
            stack_last_files(vessel, variables, base_dir)
            stack_last_files(vessel, variables, updated_dir)

        def partial_deriv_files(base_dir, updated_dir, del_dir, base_file, updated_file, vaso, parametro):
            """
            Subtrai os arquivos empilhados do base_dir dos arquivos empilhados do updated_dir,
            divide pela diferença do parâmetro atualizado e salva o resultado no del_dir.
            """
            vessels = ["vaso1", "vaso2", "vaso3"]

            # Carrega os valores do parâmetro L dos arquivos YAML
            def load_param_value(yaml_file, vaso, parametro):
                """Carrega o valor do parâmetro especificado para um determinado vaso."""
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}

                for item in yaml_data.get("network", []):
                    if item.get("label") == vaso:
                        return item.get(parametro)

                print(f"⚠️ Parâmetro '{parametro}' não encontrado para o vaso '{vaso}'.")
                return None

            # Obtém os valores de L nos arquivos YAML
            value_base = load_param_value(base_file, vaso, parametro)
            value_updated = load_param_value(updated_file, vaso, parametro)

            if value_base is None or value_updated is None:
                print("❌ Não foi possível obter os valores do parâmetro. Abortando cálculo.")
                return

            # Calcula a diferença do parâmetro
            delta_value = value_updated - value_base
            print(f"📏 Diferença do parâmetro '{parametro}': {value_updated} - {value_base} = {delta_value}")

            if delta_value == 0:
                print("⚠️ Diferença do parâmetro é zero! Evitando divisão por zero.")
                return

            # Cria o diretório de saída se não existir
            os.makedirs(del_dir, exist_ok=True)

            for vessel in vessels:
                base_file_path = os.path.join(base_dir, f"{vessel}_stacked.last")
                updated_file_path = os.path.join(updated_dir, f"{vessel}_stacked.last")
                del_file_path = os.path.join(del_dir, f"{vessel}_del_{parametro}_delta={delta_value}.last")

                # Verifica se ambos os arquivos existem
                if not os.path.exists(base_file_path) or not os.path.exists(updated_file_path):
                    print(f"❌ Arquivos para {vessel} não encontrados! Pulando...")
                    continue

                # Carrega os arquivos .last
                base_data = np.loadtxt(base_file_path)
                updated_data = np.loadtxt(updated_file_path)

                # Verifica se as dimensões são compatíveis
                if base_data.shape != updated_data.shape:
                    print(f"⚠️ Dimensões incompatíveis para {vessel}: {base_data.shape} vs {updated_data.shape}")
                    continue

                # Realiza a subtração e divisão pelo delta_L
                del_data = (updated_data - base_data) / delta_value

                # Salva o resultado
                np.savetxt(del_file_path, del_data, fmt="%.6e")
                print(f"✅ Arquivo de diferença normalizada salvo: {del_file_path}")

        # Caminho para o diretório de diferenças
        del_dir = os.path.join(openBF_dir, f"partial_deriv_{parametro}")

        # Chama a função com o parâmetro 'L'
        partial_deriv_files(base_dir, updated_dir, del_dir, base_file, output_file, vaso, parametro)

        # Plots the simulation output graphs and saves them
        def plot_openBF(data_dir):
            vessels = ["vaso1", "vaso2", "vaso3"]
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
                    #   "P": [df_vaso1_P, df_vaso2_P, df_vaso3_P],
                    #   "Q": [df_vaso1_Q, df_vaso2_Q, df_vaso3_Q],
                    #   "A": [df_vaso1_A, df_vaso2_A, df_vaso3_A],
                    #   "u": [df_vaso1_u, df_vaso2_u, df_vaso3_u]
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


# Exemplo de uso
if __name__ == "__main__":

    base_file = "C:/Users/User/Documents/problema_inverso_results_openbf/problema_inverso_Automatizado.yaml"
    output_file = "C:/Users/User/Documents/problema_inverso_results_openbf/resultado.yaml"
    openBF_dir = "C:/Users/User/Documents/problema_inverso_results_openbf"

    updater = OPENBF_Jacobean(base_file, output_file, openBF_dir)

    # Exemplo: Atualizar o vaso X no parâmetro Y para um novo valor Z
    updater.update_yaml("vaso1", "R0", 0.026)
