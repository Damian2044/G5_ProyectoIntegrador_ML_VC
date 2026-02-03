import sys
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dataframe_image as dfi
from pathlib import Path

# Agregar el directorio padre al path para importar el módulo de clustering
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clusteringRestricciones import CluORT


# ============================================================================
# CONFIGURACIÓN: Selecciona qué características analizar
# ============================================================================
CARACTERISTICAS_A_ANALIZAR = [
    'momentos',
    'sift',
    'hog',
    'embeddings'  
]
# ============================================================================




def obtenerDatasetsProcesados():
    """
    Obtiene la lista de carpetas de datasets procesados.
    
    Args:
        None
    
    Returns:
        dict: Diccionario con estructura {nombre_dataset: Path_a_carpeta}
    """
    rutaResultados = Path(__file__).parent.parent.parent / "procesador_imagenes" / "results"
    print(f"✓ Ruta de resultados: {rutaResultados}")
    
    datasets = {}
    if rutaResultados.exists():
        for carpeta in rutaResultados.iterdir():
            if carpeta.is_dir() and carpeta.name not in ['imagenes_procesadas', '__pycache__']:
                datasets[carpeta.name] = carpeta
    
    return datasets


def cargarCaracteristicas(rutaDataset, tipoCaracteristica):
    """
    Carga características de un CSV específico, extrayendo automáticamente etiqueta y columnas numéricas.
    
    Args:
        rutaDataset (Path): Ruta a la carpeta del dataset
        tipoCaracteristica (str): Tipo de característica ('momentos', 'momentos_hu', 'momentos_zernike', 'sift', 'hog', 'embeddings')
    
    Returns:
        tuple: (datos_features: ndarray, etiquetas: ndarray, columnas_features: list)
    """
    mapeoArchivos = {
        'momentos': 'momentos.csv',
        'momentos_hu': 'momentos_hu.csv',
        'momentos_zernike': 'momentos_zernike.csv',
        'sift': 'sift.csv',
        'hog': 'hog.csv',
        'embeddings': 'embeddings.csv'
    }
    
    if tipoCaracteristica not in mapeoArchivos:
        raise ValueError(f"Tipo de característica no válido: {tipoCaracteristica}")
    
    archivoCSV = rutaDataset / mapeoArchivos[tipoCaracteristica]
    
    if not archivoCSV.exists():
        print(f"⚠ Archivo no encontrado: {archivoCSV}")
        return None, None, None
    
    # Leer CSV
    df = pd.read_csv(archivoCSV)
    
    # Extraer etiquetas
    etiquetas = df['etiqueta'].values if 'etiqueta' in df.columns else np.array([])
    
    # Extraer columnas numéricas (excluyendo rutaImagen, etiqueta, estado, tamanoVector)
    columnasExcluir = {'rutaImagen', 'etiqueta', 'estado', 'tamanoVector'}
    columnasNumericas = [col for col in df.columns 
                         if col not in columnasExcluir and pd.api.types.is_numeric_dtype(df[col])]
    
    # Ordenar columnas numéricas por nombre/índice
    columnasNumericas = sorted(columnasNumericas, key=lambda x: (
        # Extraer número del nombre de columna si existe
        int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'),
        x
    ))
    
    # Extraer features
    datosFeatures = df[columnasNumericas].values
    
    return datosFeatures, etiquetas, columnasNumericas


def listarCaracteristicasDisponibles(rutaDataset):
    """
    Lista las características disponibles en una carpeta de dataset.
    
    Args:
        rutaDataset (Path): Ruta a la carpeta del dataset
    
    Returns:
        dict: Diccionario con {tipo_característica: archivo.csv}
    """
    caracteristicasDisponibles = {}
    mapeoArchivos = {
        'momentos': 'momentos.csv',
        'momentos_hu': 'momentos_hu.csv',
        'momentos_zernike': 'momentos_zernike.csv',
        'sift': 'sift.csv',
        'hog': 'hog.csv',
        'embeddings': 'embeddings.csv'
    }
    
    for tipo, archivo in mapeoArchivos.items():
        rutaArchivo = rutaDataset / archivo
        if rutaArchivo.exists():
            caracteristicasDisponibles[tipo] = rutaArchivo
    
    return caracteristicasDisponibles


def prepararDirectorioResultados():
    """
    Prepara el directorio de resultados, borrándolo si existe.
    
    Args:
        None
    
    Returns:
        Path: Ruta del directorio de resultados creado.
    """
    dirResultados = Path(__file__).parent.parent / "resultados_imagenes"
    
    if dirResultados.exists():
        shutil.rmtree(dirResultados)
        print(f"✓ Directorio existente eliminado: {dirResultados}")
    
    dirResultados.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directorio creado: {dirResultados}")
    
    return dirResultados


def ejecutarClusteringCluORT(datosFeatures, etiquetas):
    """
    Ejecuta clustering CluORT con restricciones de tamaño.
    
    Args:
        datosFeatures (ndarray): Matriz de características [n_samples x n_features]
        etiquetas (ndarray): Etiquetas reales [n_samples]
    
    Returns:
        dict: Métricas y resultados del clustering
    """
    # Aleatorizar datos con semilla fija para reproducibilidad
    from sklearn.utils import shuffle
    datosFeatures, etiquetas = shuffle(datosFeatures, etiquetas, random_state=44)

    # Calcular restricciones de tamaño por clase
    etiquetasUnicas, tamaniosMaximos = np.unique(etiquetas, return_counts=True)
    numClusters = len(etiquetasUnicas)
    tamaniosMaximos = tamaniosMaximos.tolist()
    
    # Inicializar CluORT:
    # - metricasAproximadas=True  -> en vivo usa aproximadas (rápido)
    # - guardarPuntos=True        -> guarda puntos para poder calcular las reales al final
    clustering = CluORT(
        numClusters=numClusters,
        tamaniosMaximos=tamaniosMaximos,
        metricasAproximadas=True,
        escalarDatos=True,
        guardarPuntos=True,
    )
    
    # Procesar todos los puntos en orden (streaming online)
    ultimoResultado = None
    for i, (punto, etiqueta) in enumerate(zip(datosFeatures, etiquetas)):
        ultimoResultado = clustering.asignarPunto(punto, etiquetaReal=etiqueta)
        
        # Mostrar progreso cada 100 puntos (métricas aproximadas en vivo)
        if (i + 1) % 100 == 0:
            aprox = clustering.obtenerResumenFinal(usarAproximadas=True)
            sil_ap = aprox['metricasInternas']['silueta']
            dunn_ap = aprox['metricasInternas']['dunn']
            print(f"\r      Procesados: {i + 1}/{len(datosFeatures)} | Sil(aprox): {sil_ap:.4f} | Dunn(aprox): {dunn_ap:.4f}", end="", flush=True)
    
    print()  # Nueva línea después del progreso
    
    # Al final: métricas REALES (exactas) usando los puntos guardados
    resumen_final = clustering.obtenerResumenFinal(usarAproximadas=False)
    metricasInternas = resumen_final['metricasInternas']
    metricasExternas = resumen_final['metricasExternas']
    distribucion = resumen_final['distribucion']
    
    return {
        'silueta': metricasInternas['silueta'],
        'dunn': metricasInternas['dunn'],
        'ari': metricasExternas['ari'],
        'ami': metricasExternas['ami'],
        'nmi': metricasExternas['nmi'],
        'distribucion': distribucion,
        'tamaniosActuales': resumen_final['tamaniosActuales'],
        'tamaniosMaximos': resumen_final['tamaniosMaximos'],
    }


def formatearDistribucion(distribucion):
    """
    Formatea la distribución de clusters para mostrar en tabla.
    
    Args:
        distribucion (dict): Distribución de etiquetas por cluster
    
    Returns:
        str: String formateado para tabla HTML
    """
    clusters = []
    for clusterKey in sorted(distribucion.keys()):
        clusterIdx = int(clusterKey.split('_')[1])
        etiquetas = distribucion[clusterKey]['etiquetasReales']
        
        # Formatear etiquetas
        etiquetasStr = ' '.join([f"{label}:{count}" 
                                  for label, count in sorted(etiquetas.items()) 
                                  if label != 'sinEtiqueta'])
        
        clusters.append(f"C{clusterIdx}:{etiquetasStr}")
    
    return '<br>'.join(clusters)


def formatearCapacidad(tamaniosActuales, tamaniosMaximos):
    """Formatea la capacidad de cada cluster como actual/maximo.

    Args:
        tamaniosActuales (Sequence): Tamaños actuales de cada cluster.
        tamaniosMaximos (Sequence): Tamaños máximos permitidos por cluster.

    Returns:
        str: Texto HTML con formato "C0: actual/max" por línea.
    """
    if tamaniosActuales is None or tamaniosMaximos is None:
        return ""

    try:
        longitud = min(len(tamaniosActuales), len(tamaniosMaximos))
    except TypeError:
        return ""

    partes = []
    for indiceCluster in range(longitud):
        actual = int(tamaniosActuales[indiceCluster])
        maximo = int(tamaniosMaximos[indiceCluster]) if tamaniosMaximos[indiceCluster] is not None else 0
        partes.append(f"C{indiceCluster}: {actual}/{maximo}")

    return '<br>'.join(partes)


def crearTablaResultados(resultados, dirResultados, nombreTabla='tabla_resultados'):
    """
    Crea y guarda tabla de resultados como PNG.
    
    Args:
        resultados (list): Lista de diccionarios con resultados
        dirResultados (Path): Directorio donde guardar
        nombreTabla (str): Nombre del archivo sin extensión
    
    Returns:
        pd.DataFrame: DataFrame con los resultados
    """
    df = pd.DataFrame(resultados)

    # Redondear métricas
    for col in ['silueta', 'dunn', 'ari', 'ami', 'nmi']:
        if col in df.columns:
            df[col] = df[col].round(3)

    # Formatear distribución
    if 'distribucion' in df.columns:
        df['distribucion'] = df['distribucion'].apply(formatearDistribucion)

    # Reemplazar <br> por saltos de línea para dataframe_image (como en Iris)
    if 'distribucion' in df.columns:
        df['distribucion'] = df['distribucion'].str.replace('<br>', '\n', regex=False)
    if 'capacidad' in df.columns:
        df['capacidad'] = df['capacidad'].str.replace('<br>', '\n', regex=False)

    # Reordenar columnas
    columnasOrden = ['dataset', 'caracteristica', 'muestras', 'features', 'clases',
                     'silueta', 'dunn', 'ari', 'ami', 'nmi', 'distribucion', 'capacidad']
    columnasExistentes = [col for col in columnasOrden if col in df.columns]
    df = df[columnasExistentes]

    # Aplicar estilo tipo "booktabs" similar al de Iris
    styled = df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '9pt',
        'font-family': 'Arial',
        'white-space': 'pre-wrap'
    }).set_table_styles([
        # Encabezado: línea superior e inferior gruesas, sin bordes verticales
        {'selector': 'thead th', 'props': [
            ('border-top', '2px solid black'),
            ('border-bottom', '2px solid black'),
            ('border-left', 'none'),
            ('border-right', 'none'),
            ('background-color', 'white'),
            ('font-weight', 'bold')
        ]},
        # Celdas del cuerpo: solo líneas horizontales finas
        {'selector': 'tbody td', 'props': [
            ('border-top', '0.5px solid #cccccc'),
            ('border-left', 'none'),
            ('border-right', 'none')
        ]},
        # Última fila: línea inferior gruesa
        {'selector': 'tbody tr:last-child td', 'props': [
            ('border-bottom', '2px solid black')
        ]},
        # Sin bordes laterales externos
        {'selector': '', 'props': [
            ('border-left', 'none'),
            ('border-right', 'none')
        ]}
    ]).hide(axis='index')

    archivoPNG = dirResultados / f"{nombreTabla}.png"
    dfi.export(styled, str(archivoPNG), dpi=300)
    print(f"✓ Tabla guardada: {archivoPNG.name}")
    
    # Guardar CSV
    archivoCSV = dirResultados / f"{nombreTabla}.csv"
    df.to_csv(archivoCSV, index=False)
    print(f"✓ CSV guardado: {archivoCSV.name}")
    
    return df


def graficarDistribucionesImagenes(resultadosGlobales, dirResultados):
    """Genera gráficos de barras apiladas con la distribución de clases por cluster
    para cada dataset y característica (similar a las gráficas de Iris).

    Para cada dataset se crea una figura con subplots: uno por tipo de característica.

    Args:
        resultadosGlobales (list): Lista de diccionarios con resultadosGlobales
            (cada uno con claves: dataset, caracteristica, distribucion, silueta, dunn, ari, ami, nmi)
        dirResultados (Path): Directorio donde guardar los gráficos PNG
    """
    if not resultadosGlobales:
        return

    # Agrupar resultados por dataset
    conjuntosDatos = {}
    for resultado in resultadosGlobales:
        nombreDataset = resultado['dataset']
        conjuntosDatos.setdefault(nombreDataset, []).append(resultado)

    for nombreDataset, resultadosPorDataset in conjuntosDatos.items():
        # Ordenar características en el orden deseado: momentos, sift, hog, embeddings
        ordenCaracteristicas = ['momentos', 'sift', 'hog', 'embeddings']
        resultadosPorDataset = sorted(
            resultadosPorDataset,
            key=lambda x: ordenCaracteristicas.index(x['caracteristica'])
            if x['caracteristica'] in ordenCaracteristicas
            else len(ordenCaracteristicas) + 1,
        )

        numeroConfiguraciones = len(resultadosPorDataset)

        # Obtener todas las clases reales que aparecen en las distribuciones
        conjuntoClases = set()
        for resultado in resultadosPorDataset:
            distribucion = resultado['distribucion']
            for claveCluster, datosCluster in distribucion.items():
                for clase in datosCluster['etiquetasReales'].keys():
                    if clase != 'sinEtiqueta':
                        conjuntoClases.add(clase)

        clasesUnicas = sorted(list(conjuntoClases))
        if not clasesUnicas:
            continue

        numClases = len(clasesUnicas)

        # Configurar figura: subplots por característica (2 columnas)
        numeroColumnas = 2
        numeroFilas = (numeroConfiguraciones + numeroColumnas - 1) // numeroColumnas
        figura, ejes = plt.subplots(numeroFilas, numeroColumnas, figsize=(16, 4 * numeroFilas))
        ejes = ejes.flatten() if numeroConfiguraciones > 1 else [ejes]

        # Paleta de colores específica para imágenes (distinta a la de Iris)
        # Usamos tonos suaves de "Set2" y, si hay muchas clases, una variante
        # más amplia basada en "tab20b".
        if numClases <= 8:
            mapaColores = plt.colormaps.get_cmap('Set2')
        elif numClases <= 20:
            mapaColores = plt.colormaps.get_cmap('tab20b')
        else:
            mapaColores = plt.colormaps.get_cmap('gist_ncar')
        coloresClases = [mapaColores(i) for i in np.linspace(0, 0.9, numClases)]

        for indiceConfiguracion, resultado in enumerate(resultadosPorDataset):
            eje = ejes[indiceConfiguracion]

            diccionarioDistribucion = resultado['distribucion']

            # Número de clusters según la distribución
            indicesClusters = [int(clave.split('_')[1]) for clave in diccionarioDistribucion.keys()]
            if not indicesClusters:
                eje.axis('off')
                continue
            numClusters = max(indicesClusters) + 1

            # Matriz clusters x clases
            matrizDistribucion = np.zeros((numClusters, numClases))

            for claveCluster, datosCluster in diccionarioDistribucion.items():
                indiceCluster = int(claveCluster.split('_')[1])
                etiquetasReales = datosCluster['etiquetasReales']

                for clase, cantidad in etiquetasReales.items():
                    if clase != 'sinEtiqueta' and clase in clasesUnicas:
                        indiceClase = clasesUnicas.index(clase)
                        matrizDistribucion[indiceCluster, indiceClase] = cantidad

            posicionesY = np.arange(numClusters)
            acumulado = np.zeros(numClusters)

            for indiceClase, clase in enumerate(clasesUnicas):
                valores = matrizDistribucion[:, indiceClase]

                barras = eje.barh(
                    posicionesY,
                    valores,
                    left=acumulado,
                    color=coloresClases[indiceClase],
                    alpha=0.85,
                    edgecolor='black',
                    linewidth=0.8,
                )

                for indiceBarra, (barra, valor) in enumerate(zip(barras, valores)):
                    if valor > 0:
                        posicionX = acumulado[indiceBarra] + valor / 2
                        eje.text(
                            posicionX,
                            barra.get_y() + barra.get_height() / 2.0,
                            f'{int(valor)}',
                            ha='center',
                            va='center',
                            fontsize=8,
                            fontweight='bold',
                            color='black',
                            bbox=dict(
                                boxstyle='round,pad=0.2',
                                facecolor='white',
                                edgecolor='none',
                                alpha=0.7,
                            ),
                        )

                acumulado += valores

            # Totales por cluster (tamaño ACTUAL) y anotación actual/máximo
            totalesPorCluster = matrizDistribucion.sum(axis=1)

            # Tamaños máximos de cada cluster (restricciones del clustering online)
            tamaniosMaximosClusters = resultado.get('tamaniosMaximos', None)
            if isinstance(tamaniosMaximosClusters, (list, tuple, np.ndarray)):
                tamaniosMaximosClusters = np.array(tamaniosMaximosClusters, dtype=int)
            else:
                tamaniosMaximosClusters = None

            for indiceCluster, totalCluster in enumerate(totalesPorCluster):
                if totalCluster <= 0:
                    continue

                if (
                    tamaniosMaximosClusters is not None
                    and indiceCluster < len(tamaniosMaximosClusters)
                    and tamaniosMaximosClusters[indiceCluster] > 0
                ):
                    denominadorCluster = int(tamaniosMaximosClusters[indiceCluster])
                else:
                    denominadorCluster = int(totalCluster)

                eje.text(
                    totalCluster * 1.01,
                    posicionesY[indiceCluster],
                    f"{int(totalCluster)}/{denominadorCluster}",
                    va='center',
                    ha='left',
                    fontsize=8,
                    fontweight='bold',
                    color='#333333',
                )

            eje.set_yticks(posicionesY)
            eje.set_yticklabels([f'C{indice}' for indice in range(numClusters)], fontsize=10)
            eje.set_xlabel('Cantidad de puntos', fontsize=10, fontweight='bold')
            eje.set_ylabel('Cluster', fontsize=10, fontweight='bold')

            # Usar el máximo total de puntos por cluster para el límite del eje X
            valorMaximo = matrizDistribucion.sum(axis=1).max()
            eje.set_xlim(0, valorMaximo * 1.15 if valorMaximo > 0 else 1)
            eje.invert_yaxis()
            eje.grid(True, axis='x', alpha=0.3, linestyle='--')

            # Título con nombre de característica y métricas principales
            silueta = resultado.get('silueta', 0.0)
            dunn = resultado.get('dunn', 0.0)
            ari = resultado.get('ari', 0.0)
            ami = resultado.get('ami', 0.0)
            nmi = resultado.get('nmi', 0.0)
            titulo = (
                f"{resultado['caracteristica']}  |  Sil: {silueta:.2f}  |  "
                f"D: {dunn:.2f}  |  ARI: {ari:.2f}  |  "
                f"AMI: {ami:.2f}  |  NMI: {nmi:.2f}"
            )
            eje.text(
                0.5,
                1.12,
                titulo,
                transform=eje.transAxes,
                fontsize=9,
                fontweight='bold',
                va='bottom',
                ha='center',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='#E5F5FF',
                    edgecolor='#4E79A7',
                    linewidth=1.2,
                    alpha=0.95,
                ),
            )

        # Ocultar subplots vacíos
        for indice in range(numeroConfiguraciones, len(ejes)):
            ejes[indice].set_visible(False)

        # Leyenda de clases común
        elementosLeyenda = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=coloresClases[i],
                alpha=0.85,
                edgecolor='black',
                linewidth=1,
            )
            for i in range(numClases)
        ]
        figura.legend(
            elementosLeyenda,
            [str(c) for c in clasesUnicas],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(numClases, 6),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=False,
        )

        figura.suptitle(
            f"{nombreDataset} - Distribución de clases por cluster y característica",
            fontsize=14,
            fontweight='bold',
            y=0.96,
        )
        figura.tight_layout(rect=[0, 0, 1, 0.93])

        archivoGrafico = dirResultados / f"distribucion_clases_{nombreDataset}.png"
        figura.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de distribución guardado: {archivoGrafico.name}")
        plt.close(figura)

def graficarRadarMetricasEstilizado(resultadosGlobales, dirResultados):
    """Genera una gráfica radial (radar) por dataset con estilo mejorado.

    - Cada eje radial es una métrica: Silueta, Dunn, ARI, AMI, NMI.
    - Cada polígono es una característica: momentos, sift, hog, embeddings, etc.
        - Las métricas se normalizan con min–max por dataset a [0, 1] para
            compararlas entre características dentro de cada conjunto de datos.
    """
    if not resultadosGlobales:
        return

    nombresMetricas = ['silueta', 'dunn', 'ari', 'ami', 'nmi']
    etiquetasMetricas = ['Silueta', 'Dunn', 'ARI', 'AMI', 'NMI']
    numeroMetricas = len(nombresMetricas)

    # Agrupar por dataset
    conjuntosDatos = {}
    for resultado in resultadosGlobales:
        nombreDataset = resultado['dataset']
        conjuntosDatos.setdefault(nombreDataset, []).append(resultado)

    for nombreDataset, resultadosPorDataset in conjuntosDatos.items():
        listaCaracteristicas = sorted({r['caracteristica'] for r in resultadosPorDataset})
        if not listaCaracteristicas:
            continue

        numeroCaracteristicas = len(listaCaracteristicas)

        # Construir matriz de métricas
        matrizMetricas = np.zeros((numeroCaracteristicas, numeroMetricas), dtype=float)
        for indiceCaracteristica, nombreCaracteristica in enumerate(listaCaracteristicas):
            resultadoCaracteristica = next(
                (r for r in resultadosPorDataset if r['caracteristica'] == nombreCaracteristica),
                None,
            )
            if resultadoCaracteristica is None:
                continue

            for indiceMetrica, nombreMetrica in enumerate(nombresMetricas):
                valor = float(resultadoCaracteristica.get(nombreMetrica, 0.0))
                matrizMetricas[indiceCaracteristica, indiceMetrica] = valor

        # Normalizar columnas a [0, 1] usando min–max POR DATASET
        matrizNormalizada = np.zeros_like(matrizMetricas)
        for indiceMetrica, nombreMetrica in enumerate(nombresMetricas):
            columna = matrizMetricas[:, indiceMetrica]
            minimo = float(np.min(columna))
            maximo = float(np.max(columna))
            if maximo - minimo > 1e-8:
                matrizNormalizada[:, indiceMetrica] = (columna - minimo) / (maximo - minimo)
            else:
                matrizNormalizada[:, indiceMetrica] = 0.5

        # Ajuste visual: comprimir rango a [0.2, 1.0] para que los valores
        # pequeños no queden pegados al centro y se aprecien mejor
        matrizNormalizada = 0.2 + 0.8 * matrizNormalizada

        # Ángulos (5 ejes) y cierre para los polígonos
        angulosBase = np.linspace(0, 2 * np.pi, numeroMetricas, endpoint=False)
        angulos = np.concatenate([angulosBase, [angulosBase[0]]])

        figura, eje = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        figura.patch.set_facecolor('#fafbfc')
        eje.set_facecolor('#ffffff')

        eje.yaxis.grid(True, color='#e8eaed', linestyle='--', linewidth=0.5, alpha=0.6)
        eje.xaxis.grid(True, color='#dadce0', linestyle='-', linewidth=0.8, alpha=0.5)

        eje.set_ylim(0, 1.05)  # Pequeño margen superior
        nivelesRadio = [0.2, 0.4, 0.6, 0.8, 1.0]
        eje.set_yticks(nivelesRadio)
        eje.set_yticklabels(
            ['0.2', '0.4', '0.6', '0.8', '1.0'], 
            fontsize=9, 
            color='#5f6368',
            fontweight='500'
        )

        eje.set_xticks(angulosBase)
        eje.set_xticklabels(
            etiquetasMetricas,
            fontsize=13,
            fontweight='600',
            color='#202124',
        )

        # Paleta de colores específica para características
        coloresPersonalizados = [
            '#1a73e8', '#ea4335', '#34a853', '#fbbc04', '#ff6d00',
            '#9334e6', '#00bfa5', '#d50000', '#6200ea', '#0091ea'
        ]
        
        if numeroCaracteristicas <= len(coloresPersonalizados):
            coloresCaracteristicas = coloresPersonalizados[:numeroCaracteristicas]
        else:
            mapaColores = plt.colormaps.get_cmap('tab20')
            coloresCaracteristicas = [
                mapaColores(i) for i in np.linspace(0, 0.95, numeroCaracteristicas)
            ]

        for indiceCaracteristica, nombreCaracteristica in enumerate(listaCaracteristicas):
            valores = matrizNormalizada[indiceCaracteristica]
            valoresCerrados = np.concatenate([valores, [valores[0]]])

            color = coloresCaracteristicas[indiceCaracteristica]

            eje.plot(
                angulos,
                valoresCerrados,
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=8,
                markeredgewidth=1.5,
                markeredgecolor='white',
                label=nombreCaracteristica,
                alpha=0.9,
                zorder=3
            )
            
            eje.fill(
                angulos,
                valoresCerrados,
                color=color,
                alpha=0.12,
                zorder=2
            )

        figura.suptitle(
            nombreDataset,
            fontsize=18,
            fontweight='bold',
            color='#202124',
            y=0.98,
        )

        ncol = 2 if numeroCaracteristicas > 6 else 1
        leyenda = eje.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.8,
            ncol=ncol,
            edgecolor='#dadce0',
            facecolor='white',
        )
        leyenda.get_frame().set_linewidth(1.2)
        leyenda.get_frame().set_alpha(0.95)

        archivoRadar = dirResultados / f"radial_metricas_{nombreDataset}.png"
        figura.tight_layout(rect=[0, 0.05, 1, 0.95])  # Ajuste centrado con leyenda inferior
        figura.savefig(archivoRadar, dpi=400, bbox_inches='tight', 
                      facecolor='white', edgecolor='none')
        print(f"✓ Gráfica radial de métricas guardada: {archivoRadar.name}")
        plt.close(figura)


def main():
    """
    Función principal que orquesta el pipeline de experimentación con imágenes.
    Carga datasets de imágenes, aplica clustering CluORT y STREAMKMeans, y genera visualizaciones.
    
    Args:
        None
    
    Returns:
        None (genera archivos de resultados)
    """
    print(f"\n{'='*100}")
    print(f"EXPERIMENTOS DE CLUSTERING CON IMÁGENES")
    print(f"{'='*100}")
    
    # 1. Preparar directorio de resultados
    dirResultados = prepararDirectorioResultados()
    
    # 2. Obtener datasets procesados
    conjuntosDatos = obtenerDatasetsProcesados()
    
    if not conjuntosDatos:
        print("⚠ No se encontraron datasets en la carpeta de resultados")
        return
    
    print(f"\n✓ Datasets encontrados: {list(conjuntosDatos.keys())}")
    print(f"✓ Características configuradas para análisis: {CARACTERISTICAS_A_ANALIZAR}")
    
    # 3. Ejecutar clustering para cada combinación dataset + característica
    resultadosGlobales = []
    
    for nombreDataset, rutaDataset in conjuntosDatos.items():
        print(f"\n{'─'*100}")
        print(f"Dataset: {nombreDataset}")
        print(f"{'─'*100}")
        
        caracteristicas = listarCaracteristicasDisponibles(rutaDataset)
        
        if not caracteristicas:
            print(f"⚠ No se encontraron características en: {rutaDataset}")
            continue
        
        # Procesar cada característica seleccionada
        for tipoCaracteristica in CARACTERISTICAS_A_ANALIZAR:
            if tipoCaracteristica not in caracteristicas:
                print(f"  ✗ {tipoCaracteristica}: No disponible")
                continue
            
            print(f"\n  Procesando {tipoCaracteristica}...")
            
            # Cargar datos
            datosFeatures, etiquetas, columnasNumericas = cargarCaracteristicas(rutaDataset, tipoCaracteristica)
            
            if datosFeatures is None or len(datosFeatures) == 0:
                print(f"    ⚠ Error al cargar datos")
                continue
            
            numMuestras = len(datosFeatures)
            numFeatures = len(columnasNumericas)
            numClases = len(np.unique(etiquetas))
            
            print(f"    - Muestras: {numMuestras}, Features: {numFeatures}, Clases: {numClases}")
            
            # Ejecutar clustering CluORT
            print(f"    - Ejecutando CluORT...", end=" ")
            try:
                resultadoClustering = ejecutarClusteringCluORT(datosFeatures, etiquetas)
                print("✓")
                
                # Mostrar métricas
                print(f"      Silueta: {resultadoClustering['silueta']:.4f} | "
                      f"Dunn: {resultadoClustering['dunn']:.4f} | "
                      f"ARI: {resultadoClustering['ari']:.4f}")
                
                # Almacenar resultados
                capacidadTexto = formatearCapacidad(
                    resultadoClustering.get('tamaniosActuales'),
                    resultadoClustering.get('tamaniosMaximos'),
                )

                resultadosGlobales.append({
                    'dataset': nombreDataset,
                    'caracteristica': tipoCaracteristica,
                    'muestras': numMuestras,
                    'features': numFeatures,
                    'clases': numClases,
                    'silueta': resultadoClustering['silueta'],
                    'dunn': resultadoClustering['dunn'],
                    'ari': resultadoClustering['ari'],
                    'ami': resultadoClustering['ami'],
                    'nmi': resultadoClustering['nmi'],
                    'distribucion': resultadoClustering['distribucion'],
                    'tamaniosMaximos': resultadoClustering['tamaniosMaximos'],
                    'capacidad': capacidadTexto,
                })
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    # 4. Generar tablas de resultados
    if not resultadosGlobales:
        print("\n⚠ No se generaron resultados")
        return
    
    print(f"\n{'='*100}")
    print(f"GENERANDO TABLAS DE RESULTADOS")
    print(f"{'='*100}")
    
    # Tabla global con todos los resultados
    dfGlobal = crearTablaResultados(resultadosGlobales, dirResultados, 'resultados_clustering_global')
    
    # Tablas separadas por dataset
    for nombreDataset in conjuntosDatos.keys():
        resultadosDataset = [r for r in resultadosGlobales if r['dataset'] == nombreDataset]
        if resultadosDataset:
            crearTablaResultados(resultadosDataset, dirResultados, f'resultados_clustering_{nombreDataset}')

    # 5. Gráficos de distribución (barras apiladas) por dataset
    print(f"\n{'='*100}")
    print(f"GENERANDO GRÁFICOS DE DISTRIBUCIÓN DE CLASES")
    print(f"{'='*100}")
    graficarDistribucionesImagenes(resultadosGlobales, dirResultados)
    
    # 6. Gráfica radial de métricas por característica y dataset
    print(f"\n{'='*100}")
    print(f"GENERANDO GRÁFICAS RADIALES DE MÉTRICAS")
    print(f"{'='*100}")
    graficarRadarMetricasEstilizado(resultadosGlobales, dirResultados)
    
    print(f"\n{'='*100}")
    print(f"EXPERIMENTOS COMPLETADOS")
    print(f"Resultados guardados en: {dirResultados}")
    print(f"{'='*100}")





if __name__ == "__main__":
    main()
