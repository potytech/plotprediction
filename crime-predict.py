import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os
import pickle
import folium

# Caminho do modelo treinado
caminho_modelo_svm = "modelo_svm.pkl"
# Caminho do vectorizer treinado
caminho_vectorizer = "vectorizer.pkl"

# Caminho do arquivo CSV com dados geográficos dos bairros de Teresina
caminho_bairros_teresina = "bairros.csv"

# Função para carregar o modelo e o vectorizer
def carregar_modelo_e_vectorizer():
    if not os.path.exists(caminho_modelo_svm):
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo_svm}")

    if not os.path.exists(caminho_vectorizer):
        raise FileNotFoundError(f"Vectorizer não encontrado: {caminho_vectorizer}")

    with open(caminho_modelo_svm, "rb") as f:
        modelo = pickle.load(f)

    with open(caminho_vectorizer, "rb") as f:
        vectorizer = pickle.load(f)

    return modelo, vectorizer

# Função para prever o tipo de crime
def prever_tipo_crime_svm(descricao, vectorizer, modelo):
    descricao_vetor = vectorizer.transform([descricao])
    predicao = modelo.predict(descricao_vetor)
    return predicao[0]

# Função para gerar relatório de frequências e plotar no mapa
def gerar_relatorio_frequencias_e_plotar_mapa(dados_preditos, dados_originais, dados_bairros_teresina):
    # Calcular total de ocorrências de cada tipo de crime
    total_por_tipo = dados_preditos["Tipo de Crime"].value_counts()

    # Calcular o total de casos analisados
    total_casos = dados_preditos.shape[0]

    # Criar um mapa de Teresina
    mapa = folium.Map(location=[-5.090, -42.803], zoom_start=12)

    # Dicionário para armazenar a quantidade de cada tipo de crime por bairro
    quantidade_por_bairro = {bairro: {} for bairro in dados_originais["Local"].unique()}

    # Adicionar marcadores para os bairros onde ocorreram crimes
    for index, row in dados_originais.iterrows():
        bairro = row["Local"]
        crime = dados_preditos.loc[index, "Tipo de Crime"]
        if crime not in quantidade_por_bairro[bairro]:
            quantidade_por_bairro[bairro][crime] = 1
        else:
            quantidade_por_bairro[bairro][crime] += 1

    # Adicionar marcadores para os bairros onde ocorreram crimes
    for bairro, quantidade in dados_originais["Local"].value_counts().items():
        # Encontrar as coordenadas do bairro
        coordenadas = dados_bairros_teresina[dados_bairros_teresina["Bairro"] == bairro][["Latitude", "Longitude"]].values[0]

        # Criar a string com as informações de ocorrências por tipo de crime
        popup_content = f"Bairro: {bairro}\nOcorrências totais: {quantidade}\n"
        if bairro in quantidade_por_bairro:
            popup_content += "\nOcorrências por tipo de crime:"
            for crime, qtd in quantidade_por_bairro[bairro].items():
                popup_content += f"\n- {crime}: {qtd}"

        # Adicionar marcador com o número de ocorrências no bairro
        folium.Marker(location=coordenadas, popup=popup_content).add_to(mapa)

    # Adicionar o total de casos analisados ao final do relatório
    relatorio = "Relatório de Frequências de Crimes\n\n"
    for tipo_crime, total in total_por_tipo.items():
        relatorio += f"Tipo de Crime: {tipo_crime}\n"
        relatorio += f"- Total de ocorrências: {total}\n"

    relatorio += f"\nTotal de casos analisados: {total_casos}\n"
    print(relatorio)

    # Salvar o mapa como um arquivo HTML
    mapa.save("mapa_crimes_teresina.html")



# Função principal
def main():
    # Carregar modelo e vectorizer
    modelo, vectorizer = carregar_modelo_e_vectorizer()

    # Ler dados do arquivo CSV
    dados_originais = pd.read_csv("ocorrencias.csv")

    # Ler dados do arquivo CSV com informações geográficas dos bairros de Teresina
    dados_bairros_teresina = pd.read_csv(caminho_bairros_teresina)

    # Prever tipo de crime para cada ocorrência
    dados_preditos = pd.DataFrame({"Tipo de Crime": [prever_tipo_crime_svm(descricao, vectorizer, modelo) for descricao in dados_originais["Descrição"]]})

    # Gerar relatório de frequências e plotar no mapa
    gerar_relatorio_frequencias_e_plotar_mapa(dados_preditos, dados_originais, dados_bairros_teresina)

if __name__ == "__main__":
    main()
