# Robust MHE Experiment

Este repositorio contiene experimentos relacionados con el uso de Estimación de Estados Robusta (MHE) utilizando la pérdida de Huber. Los datos provienen de sensores GPS e IMU, y se realizan análisis y visualizaciones para evaluar el desempeño de diferentes métodos de estimación.

## Contenido

- **GPS_MHE.ipynb**: Notebook principal que contiene el análisis y las visualizaciones.
- **herramientas.py**: Archivo con funciones auxiliares utilizadas en el notebook.
- **gps_x_y_vx_vy.npz**: Archivo con datos del GPS.
- **imu_ax_ay.npz**: Archivo con datos del IMU.

## Requisitos

- Python 3.8 o superior
- Bibliotecas necesarias:
  - numpy
  - matplotlib
  - casadi

## Uso

1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Abre el archivo `GPS_MHE.ipynb` en Jupyter Notebook o JupyterLab para explorar los experimentos.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.