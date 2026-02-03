"""
Configuración del proyecto de procesamiento de imágenes
"""

import os

# Configuraciones de datasets
CONJUNTOS_DATOS = {
    'ecommerce': {
        'nombre': 'ecommerce-products-image-dataset',
        'idKaggle': 'sunnykusawa/ecommerce-products-image-dataset',
        'tamanoObjetivo': (224, 224), # Tamaño estándar
        'categorias': None, # Todas las categorías
        'subcarpeta': None
    },
    'mechanical_tools': {
        'nombre': 'mechanical-tools-dataset',
        'idKaggle': 'salmaneunus/mechanical-tools-dataset',
        'tamanoObjetivo': (224, 224), # Tamaño estándar
        'categorias': ['pliers', 'Rope', 'Toolbox'], # Solo estas categorías
        'subcarpeta': 'Mechanical Tools Image dataset/Mechanical Tools Image dataset' # Carpeta específica
    }
}

# Algoritmos de extracción de características
EXTRACTORES_CARACTERISTICAS = ['momentos', 'sift', 'hog']

# Configuración de preprocesamiento general (aplica a todo el conjunto)
CONFIGURACION_PREPROCESAMIENTO = {
    'espacioColor': 'grayscale',
    'normalizar': True,
    'correccionAutomatica': True,
    'fondoBlanco': False,
    'guardarImagenes': True
}

# Rutas de directorios
RAIZ_PROYECTO = os.path.dirname(os.path.abspath(__file__))
DIR_SRC = os.path.join(RAIZ_PROYECTO, 'src')
DIR_CONJUNTOS = os.path.join(RAIZ_PROYECTO, 'datasets')
DIR_RESULTADOS = os.path.join(RAIZ_PROYECTO, 'results')

# Crear directorios si no existen
os.makedirs(DIR_CONJUNTOS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


# Parámetros de extracción de características
PARAMETROS_CARACTERISTICAS = {
    'momentos': {
        'ordenMaximo': 3,   # Orden máximo de momentos
        'momentosHu': 7     # 7 momentos invariantes de Hu
    },
    'sift': {
        'numeroCaracteristicas': 500,   # Número máximo de características a detectar
        'octavas': 4,       # Número de octavas
        'escalas': 5,       # Número de escalas por octava
        'sigma': 1.6,       # Sigma inicial
        'umbral': 0.03      # Umbral de contraste
    },
    'hog': {
        'orientaciones': 9,     # Número de orientaciones
        'pixelesPorCelda': (16,16),    # Tamaño de celda en píxeles
        'celdasPorBloque': (2, 2),      # Número de celdas por bloque
        'normalizacionBloque': 'L2-Hys' # Tipo de normalización del bloque
    }
}

