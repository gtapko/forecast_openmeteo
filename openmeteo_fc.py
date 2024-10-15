from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import requests
import numpy as np
import seaborn as sns
import datetime
import re
import pytz
from urllib.parse import urlparse, parse_qs

import re

class Pronostico:
    def obtener_pronostico_meteorologico(self, Coordenadas, Horas):
        if Coordenadas.startswith("http"):
            # Extraer latitud y longitud de la URL
            parsed_url = urlparse(Coordenadas)
            query_params = parse_qs(parsed_url.query)
            lat = query_params['lat'][0]
            lon = query_params['lon'][0]
        else:
            # Asumir que el formato es "lat,lon" y separarlo
            lat = Coordenadas.split(',')[0].strip()
            lon = Coordenadas.split(',')[1].strip()
    
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,relativehumidity_2m,dewpoint_2m,windspeed_10m,winddirection_10m,windgusts_10m,precipitation_probability,precipitation,cloudcover',
            'models': 'gfs_global',
            'timezone': 'auto',
            'forecast_days': 3
        }
    
        response = requests.get(url, params=params)
        data = response.json()
    
        SantiagoTZ = pytz.timezone("America/Santiago")
    
        # Convertir los datos en un DataFrame
        df = pd.DataFrame(data['hourly'])
    
        current_hour = datetime.datetime.now(SantiagoTZ).hour
    
        df = df.iloc[current_hour:current_hour + Horas]
    
        def calculate_values(row):
          x1 = round(0.297374 + (0.262 * row['relativehumidity_2m']) - (0.00982 * row['temperature_2m']), 2)
    
          # Calcular la dirección del viento
          if pd.isna(row['winddirection_10m']):
              x2 = 'X'
          else:
            wind_dir_names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
            wind_dir_bins = np.array([0, 11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75, 191.25, 213.75, 236.25, 258.75, 281.25, 303.75, 326.25, 348.75, 361])
            idx = np.digitize(row['winddirection_10m'], wind_dir_bins) - 1
            x2 = wind_dir_names[idx]
    
          row['hcfm'] = x1
          row['windname'] = x2
    
          return row
    
        df = df.apply(calculate_values, axis=1)
        df_export = df.reindex(['time', 'temperature_2m', 'dewpoint_2m', 'relativehumidity_2m', 'hcfm', 'windspeed_10m', 'windgusts_10m', 'precipitation_probability', 'precipitation', 'cloudcover', 'winddirection_10m', 'windname'], axis=1)
    
        # Definir los nombres de las direcciones del viento
        directions = list(df['windname'])
    
        # Obtener la fecha y hora de inicio y fin en hora local
        start_time = pd.to_datetime(df['time'].iloc[0]).strftime('%d-%m-%Y %H:%M')
        end_time = pd.to_datetime(df['time'].iloc[-1]).strftime('%d-%m-%Y %H:%M')
        start_time_file = pd.to_datetime(df['time'].iloc[0]).strftime('%d%m%Yx%H:%M')
        end_time_file = pd.to_datetime(df['time'].iloc[-1]).strftime('%d%m%Yx%H:%M')
        timezone_abbr = data['timezone_abbreviation']
    
        # convertir la columna de tiempo en un objeto de fecha
        time_labels = [datetime.datetime.strptime(str(t), '%Y-%m-%dT%H:%M').strftime('%d-%b %H:%M') for t in df['time']]
    
        # Pivoteamos los datos para que las columnas sean las fechas y los índices sean los tipos de datos
        variables = ['temperature_2m', 'dewpoint_2m', 'relativehumidity_2m', 'hcfm', 'windspeed_10m', 'windgusts_10m', 'precipitation_probability', 'precipitation', 'cloudcover', 'winddirection_10m']
        new_names = ['Temp(°C)', 'Pto.Rocío(°C)', "HR(%)", "HCFM(%)", "Vel.Viento(km/h)", "Ráfagas(km/h)", "Prob.Pp(%)", 'Pp(mm)', 'Nubosidad(%)', 'Dir.Viento(°)']
        df2 = (df.loc[:, df.columns!='windname']).pivot_table(columns='time', fill_value=0)
        df3 = df2.reindex(variables)
    
        data = {}
        for col in df3.columns:
            data[col] = df3[col].tolist()
        df4 = pd.DataFrame(data, index = variables)
    
    
        def hex_to_rgb(hex_colors):
            hex_value = hex_colors.lstrip('#')
            rgb_value = mcolors.to_rgb('#' + hex_value)
            return rgb_value
    
        # Definir las paletas de colores
        hcfm_colors =['#FF00C5','#A80000','#FF0000','#FFBC00','#FFFF00','#A3FF73','#4CE600','#267300','#73B2FF','#0009FF']
        hr_colors =['#FF00C5','#850200','#FF0000','#FFBC00','#FFFF00','#A3FF73','#4CE600','#267300','#73B2FF','#0035FF']
        temp_colors = ['#0035FF','#5CA2D1','#055D00','#30C407','#FFF700','#FFCC00','#FF6600','#E31A1C','#850200']
        viento_colors = ['#5CA2D1','#055D00','#229A00','#B2DF8A','#FFF700','#FFCC00','#F99D59','#E31A1C']
        dew_colors = list(reversed(temp_colors))
    
        # Definir rangos para los datos
        vientodir_bounds = [11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75, 191.25, 213.75, 236.25, 258.75, 281.25, 303.75, 326.25, 348.75, 361]
    
        # Crear la paleta de colores
        vientodir_cmap = ListedColormap(['#00FF00', '#7FFF00', '#FFFF00', '#FFA500', '#FF4500', '#FF0000', '#FF00FF', '#8B00FF', '#0000FF', '#00BFFF', '#ADD8E6', '#87CEFA', '#00FFFF', '#00FF7F', '#00C5CD', '#FFFFFF', '#00FF00'])
        nubosity_cmap = sns.color_palette(["#FFFFFF", "#F2F2F2", "#E5E5E5", "#D9D9D9", "#CCCCCC", "#BFBFBF", "#B3B3B3", "#A6A6A6", "#999999", "#8C8C8C"])
    
        # Crear el objeto normalizador
        winddir_norm = mcolors.BoundaryNorm(vientodir_bounds, vientodir_cmap.N)
    
        cm = ['RdBu_r', 'RdBu', 'BrBG', 'BrBG', 'Purples', 'BuPu', 'Blues', 'Blues', nubosity_cmap, vientodir_cmap]
        norm = ['','','','','','','','','', winddir_norm]
        f, axs = plt.subplots(10, 1, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12, 2.5), dpi=200, sharex=True)
    
        counter = 0
        for index, row in df4.iterrows():
            if counter in [2, 6, 8, 9]:
              fmt = '.0f'
            else:
              fmt = '.1f'
            if counter == 9:
              sns.heatmap(np.array([row.values]), annot=True, fmt=fmt, ax=axs[counter], cmap=cm[counter], norm=norm[counter], cbar=False, annot_kws={"size": 35 / np.sqrt(len(directions))}, square=False, cbar_kws={"shrink": .3}, linewidths=0.2, linecolor="black")
            else:
              sns.heatmap(np.array([row.values]), annot=True, fmt=fmt, ax=axs[counter], cmap=cm[counter], cbar=False, annot_kws={"size": 35 / np.sqrt(len(directions))}, square=False, cbar_kws={"shrink": .3}, linewidths=0.2, linecolor="black")
            axs[counter].set_yticklabels([new_names[counter]], fontsize=6, rotation=0)
            axs[counter].set_xticks([])
            counter += 1
    
        # Construir la lista de valores de la tabla
        table_vals = [directions]
        table_vals.append([datetime.datetime.strptime(str(t), '%Y-%m-%dT%H:%M').strftime('%d%b\n%H:%M') for t in df['time']])
        col_widths = [0.1] * len(time_labels)
    
        n_rows = 2  # Actualizar el número de filas a 2
        n_columns = len(time_labels)
    
        row_height = 1.5
        cell_text = []
    
        for row in range(len(table_vals)):
            cell_text.append(table_vals[row])
        table = axs[-1].table(cellText=cell_text, colWidths=col_widths,
                          bbox=[0,-row_height*2,1,row_height*2],
                          cellLoc = 'center', edges='closed')
    
        # Configurar la apariencia de la tabla
        table.auto_set_font_size(False)
        table.set_fontsize(28 / np.sqrt(len(directions)))
        table.scale(1, 2)
    
        # Configurar el grosor de la línea de borde
        for cell in table.get_celld().values():
            cell.set_linewidth(0.2)
    
        # Título común para los heatbars
        title = f"Pronóstico Meteorológico - Latitud: {params['latitude']}, Longitud: {params['longitude']}\n"
        title += f" Período: {start_time} to {end_time} (UTC{timezone_abbr}) - Modelo GFS"
    
        file_name = re.sub(":|-", "", f"forecast_{params['latitude']}_{params['longitude']}_{start_time_file}_{end_time_file}").replace('.','x')
        file_name +=".xlsx"
        df_export.to_excel(file_name)
    
        f.suptitle(title, fontsize=6, y=1.0)
        plt.show()
