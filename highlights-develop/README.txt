Para poner en marcha la demo de clipado es necesario tener python y conda instalados.

Se ofrecen dos opciones para crear el entorno:

conda env create -f demo_clipado.yml

ó

conda create --name Demo_clipado
source activate Demo_clipado
conda install --file requirements.txt

Después, para lanzar el módulo en streamlit se utiliza:

streamlit run streamlit_demo.py --server.maxUploadSize=1028

# DATA______________________________________________
Juventus vs Udinese 03/01/2021 ● Serie A 2020/2021
El vídeo de "1half_Juventus_Unidese.mkv" está extraído de entre los min 0:10:14-0:57:16 del enlace:
https://www.youtube.com/watch?v=xQO2tk1aghk
Sus características preprocesadas son: "features_1half_Juventus_Unidese" y ocurre un gol en el min 30:23.
