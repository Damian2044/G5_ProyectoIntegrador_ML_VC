"""
Análisis estadístico de características extraídas de imágenes.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class AnalizadorCaracteristicas:
    """Analiza estadísticas de características extraídas."""

    def __init__(self, rutaDataset):
        self.rutaDataset = Path(rutaDataset)
        self.nombreDataset = self.rutaDataset.name
        self.estadisticas = {}

    def cargarYAnalizar(self, archivo, tipoCaracteristica):
        """Carga archivo CSV y calcula estadísticas."""
        ruta = self.rutaDataset / archivo
        if not ruta.exists():
            return None

        datos = pd.read_csv(ruta)
        # Identificar columnas numéricas (excluir metadata)
        colsExcluir = ['rutaImagen', 'etiqueta', 'estado', 'tamanoVector']
        colsCaracteristicas = [c for c in datos.columns if c not in colsExcluir]
        
        if len(colsCaracteristicas) == 0:
            return None

        # Convertir a numérico y eliminar no numéricos
        datos[colsCaracteristicas] = datos[colsCaracteristicas].apply(pd.to_numeric, errors='coerce')
        colsCaracteristicas = [c for c in colsCaracteristicas if datos[c].notna().any()]
        if len(colsCaracteristicas) == 0:
            return None
        
        datosCaracteristicas = datos[colsCaracteristicas].values
        datosCaracteristicas = datos[colsCaracteristicas].values
        clases = datos['etiqueta'].unique()
        
        info = {
            'tipo': tipoCaracteristica,
            'archivo': archivo,
            'nCaracteristicas': len(colsCaracteristicas),
            'nMuestras': len(datos),
            'nClases': len(clases),
            'clases': sorted(clases.tolist()),
            'muestrasPorClase': {},
            'estadisticasGlobales': {
                'media': float(np.nanmean(datosCaracteristicas)),
                'std': float(np.nanstd(datosCaracteristicas)),
                'min': float(np.nanmin(datosCaracteristicas)),
                'max': float(np.nanmax(datosCaracteristicas)),
                'mediana': float(np.nanmedian(datosCaracteristicas)),
                'rango': float(np.nanmax(datosCaracteristicas) - np.nanmin(datosCaracteristicas))
            },
            'estadisticasPorClase': {}
        }

        # Contar muestras por clase
        for clase in clases:
            cantidad = len(datos[datos['etiqueta'] == clase])
            info['muestrasPorClase'][clase] = int(cantidad)

        # Estadísticas por clase
        for clase in clases:
            datosClase = datos[datos['etiqueta'] == clase]
            valoresClase = datosClase[colsCaracteristicas].values
            
            info['estadisticasPorClase'][clase] = {
                'nMuestras': len(datosClase),
                'media': float(np.nanmean(valoresClase)),
                'std': float(np.nanstd(valoresClase)),
                'min': float(np.nanmin(valoresClase)),
                'max': float(np.nanmax(valoresClase))
            }

        # Coeficiente de variación
        medias = np.nanmean(datosCaracteristicas, axis=0)
        desviaciones = np.nanstd(datosCaracteristicas, axis=0)
        coefVar = desviaciones / (np.abs(medias) + 1e-10)
        
        info['coefVariacion'] = float(np.nanmean(coefVar))

        return info

    def analizarDataset(self):
        """Analiza todos los archivos de características."""
        archivosCaracteristicas = [
            ('momentos.csv', '1. Momentos (24 características)'),
            ('momentos_hu.csv', '2. Momentos de Hu'),
            ('momentos_zernike.csv', '3. Momentos de Zernike'),
            ('sift.csv', '4. SIFT'),
            ('hog.csv', '5. HOG')
        ]

        for archivo, descripcion in archivosCaracteristicas:
            info = self.cargarYAnalizar(archivo, descripcion)
            if info:
                self.estadisticas[descripcion] = info

    def mostrarEstadisticas(self):
        """Muestra estadísticas en consola."""
        print("\n" + "=" * 100)
        print(f"ANÁLISIS ESTADÍSTICO - {self.nombreDataset.upper()}".center(100))
        print("=" * 100)
        
        for tipo, info in self.estadisticas.items():
            print(f"\n{tipo}".center(100))
            print("=" * 100)
            
            # Información básica
            print(f"\nCARACTERÍSTICAS GENERALES")
            print("-" * 100)
            print(f"  Número de características: {info['nCaracteristicas']}")
            print(f"  Muestras totales:          {info['nMuestras']}")
            print(f"  Número de clases:          {info['nClases']}")
            print(f"  Clases:                    {', '.join(info['clases'])}")
            
            # Distribución de muestras por clase
            print(f"\nDISTRIBUCIÓN POR CLASE")
            print("-" * 100)
            print(f"  {'Clase':<20} {'Muestras':>10} {'Porcentaje':>12}")
            print(f"  {'─'*20} {'─'*10} {'─'*12}")
            for clase in sorted(info['muestrasPorClase'].keys()):
                cantidad = info['muestrasPorClase'][clase]
                porcentaje = (cantidad / info['nMuestras']) * 100
                print(f"  {clase:<20} {cantidad:>10} {porcentaje:>11.1f}%")

            # Estadísticas globales
            eg = info['estadisticasGlobales']
            print(f"\nESTADÍSTICAS GLOBALES")
            print("-" * 100)
            print(f"  {'Métrica':<25} {'Valor':>20}")
            print(f"  {'─'*25} {'─'*20}")
            print(f"  {'Media':<25} {eg['media']:>20.4f}")
            print(f"  {'Desviación estándar':<25} {eg['std']:>20.4f}")
            print(f"  {'Mínimo':<25} {eg['min']:>20.4f}")
            print(f"  {'Máximo':<25} {eg['max']:>20.4f}")
            print(f"  {'Mediana':<25} {eg['mediana']:>20.4f}")
            print(f"  {'Rango':<25} {eg['rango']:>20.4f}")
            print(f"  {'Coef. variación':<25} {info['coefVariacion']:>20.4f}")

            # Estadísticas por clase
            print(f"\nESTADÍSTICAS POR CLASE")
            print("-" * 100)
            print(f"  {'Clase':<20} {'Media':>20} {'Desv. Estándar':>20} {'Mínimo':>20} {'Máximo':>20}")
            print(f"  {'═'*20} {'═'*20} {'═'*20} {'═'*20} {'═'*20}")
            for clase in sorted(info['estadisticasPorClase'].keys()):
                ec = info['estadisticasPorClase'][clase]
                print(f"  {clase:<20} {ec['media']:>20.4f} {ec['std']:>20.4f} {ec['min']:>20.4f} {ec['max']:>20.4f}")
            print("=" * 100)

    def generarReporte(self):
        """Genera reporte final."""
        print("\n\n" + "=" * 100)
        print("REVISIÓN DE RESULTADOS".center(100))
        print("=" * 100)

        print("\nALGORITMOS DE EXTRACCIÓN DE CARACTERÍSTICAS".center(100))
        print("─" * 100)
        algoritmos = [
            "1. Momentos de Imagen: Momentos crudos, centrales y normalizados",
            "2. Momentos de Hu: 7 invariantes de Hu (rotación, escala, traslación)",
            "3. Momentos de Zernike: Descriptores ortogonales basados en polinomios",
            "4. SIFT: Scale-Invariant Feature Transform",
            "5. HOG: Histogram of Oriented Gradients"
        ]
        for alg in algoritmos:
            print(f"  {alg}")

        print("\n\nRESUMEN DE RESULTADOS".center(100))
        print("═" * 100)
        print(f"  {'Tipo':<45} {'Características':>18} {'Muestras':>15} {'Clases':>12}")
        print(f"  {'─'*45} {'─'*18} {'─'*15} {'─'*12}")
        
        # Mantener orden de análisis
        orden_tipos = [
            '1. Momentos (24 características)',
            '2. Momentos de Hu',
            '3. Momentos de Zernike',
            '4. SIFT',
            '5. HOG'
        ]
        for tipo in orden_tipos:
            if tipo in self.estadisticas:
                info = self.estadisticas[tipo]
                print(f"  {tipo:<45} {info['nCaracteristicas']:>18} {info['nMuestras']:>15} {info['nClases']:>12}")

        print("\n\nOBSERVACIONES".center(100))
        print("═" * 100)
        
        print("\n1. DIMENSIONALIDAD DE CARACTERÍSTICAS")
        print("-" * 100)
        # Mantener orden original 1-5
        for tipo in orden_tipos:
            if tipo in self.estadisticas:
                info = self.estadisticas[tipo]
                print(f"     {tipo:<50} {info['nCaracteristicas']:>6} características")

        print("\n2. VARIABILIDAD DE DATOS")
        print("-" * 100)
        # Mantener orden original
        for tipo in orden_tipos:
            if tipo in self.estadisticas:
                info = self.estadisticas[tipo]
                coefVar = info['coefVariacion']
                nivel = "Alta" if coefVar > 1.0 else "Moderada" if coefVar > 0.5 else "Baja"
                print(f"     {tipo:<50} {nivel:12} (CV={coefVar:.4f})")

        print("\n" + "=" * 100)


def analizarCarpetaResults(rutaResults):
    """Analiza todos los datasets en carpeta results."""
    rutaResults = Path(rutaResults)
    datasets = sorted([d for d in rutaResults.iterdir() if d.is_dir() and (d / 'caracteristicas.csv').exists()], 
                     key=lambda p: p.name)
    
    for rutaDataset in datasets:
        analizador = AnalizadorCaracteristicas(rutaDataset)
        analizador.analizarDataset()
        analizador.mostrarEstadisticas()
        analizador.generarReporte()


def main():
    rutaResults = Path("backend/procesador_imagenes/results")
    analizarCarpetaResults(rutaResults)


if __name__ == '__main__':
    main()
