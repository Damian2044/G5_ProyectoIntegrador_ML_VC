"""
Gestor de resultados (versión en español con camelCase)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from config import DIR_RESULTADOS


class GestorResultados:
    """Gestiona la presentación y almacenamiento de resultados"""
    
    def __init__(self, directorioSalida: str = None):
        self.directorioSalida = Path(directorioSalida or DIR_RESULTADOS)
        self.directorioSalida.mkdir(parents=True, exist_ok=True)
    
    def guardarResultadosJson(self, resultados: Dict, nombreArchivo: str = 'resultados.json'):
        ruta = self.directorioSalida / nombreArchivo
        
        # Crear copia para agregar etiquetas sin modificar el original
        resultadosCopia = resultados.copy()
        imagenesConEtiquetas = []
        
        for img in resultados.get('imagenes', []):
            imgCopia = img.copy()
            # Extraer etiqueta del directorio padre de la imagen
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            imgCopia['etiqueta'] = etiqueta
            imagenesConEtiquetas.append(imgCopia)
        
        resultadosCopia['imagenes'] = imagenesConEtiquetas
        
        with open(ruta, 'w') as f:
            json.dump(resultadosCopia, f, indent=2, default=str)
        print(f"Resultados JSON guardados en {ruta}")
    
    def guardarResultadosCsv(self, resultados: Dict, nombreArchivo: str = 'resultados.csv'):
        ruta = self.directorioSalida / nombreArchivo
        filas = []
        for img in resultados.get('imagenes', []):
            # Extraer etiqueta del directorio padre de la imagen
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
            }
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'momentos' in car:
                    datos = car['momentos']
                    
                    # 10 momentos crudos
                    crudos = datos.get('momentosCrudos', {})
                    for k, v in crudos.items():
                        fila[f'crudo_{k}'] = v
                    
                    # 7 momentos centrales
                    centrales = datos.get('momentosCentrales', {})
                    for k, v in centrales.items():
                        fila[f'central_{k}'] = v
                    
                    # 7 momentos normalizados
                    normalizados = datos.get('momentosNormalizados', {})
                    for k, v in normalizados.items():
                        fila[f'normalizado_{k}'] = v

                    
                    # 7 momentos de Hu
                    hu = datos.get('momentosHu', [])
                    for i, val in enumerate(hu):
                        fila[f'hu_{i}'] = val
                    
                    # Momentos de Zernike
                    zernike = datos.get('momentosZernike', [])
                    for i, val in enumerate(zernike):
                        fila[f'zernike_{i}'] = val
                    
                    fila['momentos_tamano'] = datos.get('tamanoVector', 0)
                
                if 'sift' in car:
                    fila['sift_tamano'] = car['sift'].get('tamanoVector', 0)
                if 'hog' in car:
                    fila['hog_tamano'] = car['hog'].get('tamanoVector', 0)
            filas.append(fila)
        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"Resultados CSV guardados en {ruta}")
        
        # Guardar CSVs específicos de momentos
        self._guardarCsvMomentos(resultados)
        self._guardarCsvMomentosHu(resultados)
        self._guardarCsvMomentosZernike(resultados)
        # Guardar CSVs específicos de SIFT, HOG y embeddings
        self._guardarCsvSift(resultados)
        self._guardarCsvHog(resultados)
        self._guardarCsvEmbeddings(resultados)
    
    def _guardarCsvMomentos(self, resultados: Dict):
        """Guarda CSV solo con 24 momentos (crudos, centrales, normalizados)"""
        ruta = self.directorioSalida / 'momentos.csv'
        filas = []
        for img in resultados.get('imagenes', []):
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
            }
            
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'momentos' in car:
                    datos = car['momentos']
                    
                    # 10 momentos crudos
                    crudos = datos.get('momentosCrudos', {})
                    for k, v in crudos.items():
                        fila[k] = v
                    
                    # 7 momentos centrales
                    centrales = datos.get('momentosCentrales', {})
                    for k, v in centrales.items():
                        fila[k] = v
                    
                    # 7 momentos normalizados
                    normalizados = datos.get('momentosNormalizados', {})
                    for k, v in normalizados.items():
                        fila[k] = v
            
            filas.append(fila)
        
        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"CSV de momentos (24 momentos) guardado en {ruta}")
    
    def _guardarCsvMomentosHu(self, resultados: Dict):
        """Guarda CSV solo con momentos de Hu"""
        ruta = self.directorioSalida / 'momentos_hu.csv'
        filas = []
        for img in resultados.get('imagenes', []):
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
            }
            
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'momentos' in car:
                    datos = car['momentos']
                    hu = datos.get('momentosHu', [])
                    for i, val in enumerate(hu):
                        fila[f'hu_{i}'] = val
            
            filas.append(fila)
        
        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"CSV de momentos de Hu guardado en {ruta}")
    
    def _guardarCsvMomentosZernike(self, resultados: Dict):
        """Guarda CSV solo con momentos de Zernike"""
        ruta = self.directorioSalida / 'momentos_zernike.csv'
        filas = []
        for img in resultados.get('imagenes', []):
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
            }
            
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'momentos' in car:
                    datos = car['momentos']
                    zernike = datos.get('momentosZernike', [])
                    for i, val in enumerate(zernike):
                        fila[f'zernike_{i}'] = val
            
            filas.append(fila)
        
        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"CSV de momentos de Zernike guardado en {ruta}")

    def _guardarCsvSift(self, resultados: Dict):
        """Guarda CSV con vectores SIFT completos"""
        ruta = self.directorioSalida / 'sift.csv'
        filas = []
        for idx, img in enumerate(resultados.get('imagenes', [])):
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
                'tamanoVector': None
            }
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'sift' in car:
                    datos = car['sift']
                    vec = datos.get('vectorCaracteristicas', [])
                    fila['tamanoVector'] = datos.get('tamanoVector', len(vec))
                    for i, val in enumerate(vec):
                        fila[f'v{i}'] = val
            filas.append(fila)
        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"CSV de SIFT guardado en {ruta}")

    def _guardarCsvHog(self, resultados: Dict):
        """Guarda CSV con vectores HOG completos"""
        ruta = self.directorioSalida / 'hog.csv'
        filas = []
        for idx, img in enumerate(resultados.get('imagenes', [])):
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
                'tamanoVector': None
            }
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'hog' in car:
                    datos = car['hog']
                    vec = datos.get('vectorCaracteristicas', [])
                    fila['tamanoVector'] = datos.get('tamanoVector', len(vec))
                    for i, val in enumerate(vec):
                        fila[f'v{i}'] = val
            filas.append(fila)
        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"CSV de HOG guardado en {ruta}")

    def _guardarCsvEmbeddings(self, resultados: Dict):
        """Guarda CSV con vectores de embeddings profundos completos"""
        ruta = self.directorioSalida / 'embeddings.csv'
        filas = []
        for idx, img in enumerate(resultados.get('imagenes', [])):
            rutaImagen = img.get('rutaImagen')
            etiqueta = Path(rutaImagen).parent.name if rutaImagen else 'desconocida'
            fila = {
                'rutaImagen': rutaImagen,
                'etiqueta': etiqueta,
                'estado': img.get('estado'),
                'tamanoVector': None,
            }
            if img.get('estado') == 'exito':
                caracteristicas = img.get('caracteristicas', {})
                if 'embeddings' in caracteristicas:
                    datos = caracteristicas['embeddings']
                    vector = datos.get('vectorCaracteristicas', [])
                    fila['tamanoVector'] = datos.get('tamanoVector', len(vector))
                    for i, valor in enumerate(vector):
                        fila[f'v{i}'] = valor
            filas.append(fila)

        df = pd.DataFrame(filas)
        df.to_csv(ruta, index=False)
        print(f"CSV de embeddings guardado en {ruta}")



    
    def generarReporteResumen(self, resultados: Dict) -> str:
        reporte = []
        reporte.append("=" * 60)
        reporte.append("REPORTE DE PROCESAMIENTO DE IMÁGENES")
        reporte.append("=" * 60)
        reporte.append(f"\nTotal de imágenes procesadas: {resultados.get('totalImagenes', 0)}")
        reporte.append(f"Algoritmos aplicados: {', '.join(resultados.get('algoritmos', []))}")
        
        # Mostrar estado de corrección automática
        correccionForzada = resultados.get('correccionAutomaticaForzada')
        if correccionForzada:
            reporte.append(f"Corrección automática: FORZADA")
        else:
            # Mostrar por algoritmo si está habilitada en config
            from config import CONFIGURACION_PREPROCESAMIENTO
            algosConCorreccion = [algo for algo in resultados.get('algoritmos', [])
                                 if CONFIGURACION_PREPROCESAMIENTO.get(algo, {}).get('correccionAutomatica', False)]
            if algosConCorreccion:
                reporte.append(f"Corrección automática: HABILITADA para {', '.join(algosConCorreccion)}")
            else:
                reporte.append(f"Corrección automática: DESACTIVADA")
        
        
        est = resultados.get('estadisticas', {})
        reporte.append(f"\nImágenes procesadas exitosamente: {est.get('exitosas', 0)}")
        reporte.append(f"Imágenes con error: {est.get('fallidas', 0)}")
        if est.get('tiposError'):
            reporte.append("\nTipos de error:")
            for err, cnt in est['tiposError'].items():
                reporte.append(f"  - {err}: {cnt}")
        
        # Incluir análisis de imágenes si hay corrección automática
        # Verificar si alguna imagen tiene análisis
        tieneAnalisis = any(img.get('analisisImagen') for img in resultados.get('imagenes', []))
        
        if tieneAnalisis:
            reporte.append("\n" + "=" * 60)
            reporte.append("ANÁLISIS DE IMÁGENES")
            reporte.append("=" * 60)
            
            # Agrupar análisis por condición
            analisisPorCondicion = {}
            for img in resultados.get('imagenes', []):
                if img.get('analisisImagen'):
                    analisis = img['analisisImagen']
                    condicion = analisis.get('condicion', 'desconocida')
                    if condicion not in analisisPorCondicion:
                        analisisPorCondicion[condicion] = []
                    analisisPorCondicion[condicion].append({
                        'ruta': img.get('rutaImagen'),
                        'analisis': analisis
                    })
            
            if analisisPorCondicion:
                for condicion, imagenes in analisisPorCondicion.items():
                    reporte.append(f"\n{condicion.upper()} ({len(imagenes)} imágenes):")
                    
                    # Mostrar detalles del primer ejemplo de cada condición
                    primerAnalisis = imagenes[0]['analisis']
                    reporte.append(f"  Razón: {primerAnalisis.get('razon', 'N/A')}")
                    reporte.append(f"  Transformación aplicada: {primerAnalisis.get('transformacionAplicada', 'Ninguna')}")
                    
                    # Mostrar estadísticas promedio
                    if primerAnalisis.get('estadisticas'):
                        reporte.append(f"  Estadísticas (primer ejemplo):")
                        for key, value in primerAnalisis['estadisticas'].items():
                            if isinstance(value, float):
                                reporte.append(f"    {key}: {value:.2f}")
        
        reporte.append("\n" + "=" * 60)
        return "\n".join(reporte)
    
    def imprimirResumen(self, resultados: Dict):
        resumen = self.generarReporteResumen(resultados)
        print(resumen)
        ruta = self.directorioSalida / 'reporte_resumen.txt'
        with open(ruta, 'w', encoding="utf-8") as f:
            f.write(resumen)
        print(f"\nReporte guardado en {ruta}")
    
    def guardarVectoresCaracteristicas(self, resultados: Dict, nombreArchivo: str = 'vectoresCaracteristicas.npz'):
        ruta = self.directorioSalida / nombreArchivo
        vectores = {}
        for i, img in enumerate(resultados.get('imagenes', [])):
            if img.get('estado') == 'exito':
                car = img.get('caracteristicas', {})
                if 'momentos' in car and 'vectorCaracteristicas' in car['momentos']:
                    vectores[f'momentos_{i}'] = np.array(car['momentos']['vectorCaracteristicas'])
                if 'sift' in car and 'vectorCaracteristicas' in car['sift']:
                    vectores[f'sift_{i}'] = np.array(car['sift']['vectorCaracteristicas'])
                if 'hog' in car and 'vectorCaracteristicas' in car['hog']:
                    vectores[f'hog_{i}'] = np.array(car['hog']['vectorCaracteristicas'])
        np.savez_compressed(ruta, **vectores)
        print(f"Vectores de características guardados en {ruta}")

    
