"""
Script principal de procesamiento de imágenes
"""

import sys
import os
import shutil
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocesamiento import PreprocesadorImagen
from config import CONFIGURACION_PREPROCESAMIENTO
from src.cargaDatos import GestorConjuntoDatos, CargadorKaggle
from src.utilidades import ProcesadorCaracteristicas, GestorResultados
from config import CONJUNTOS_DATOS, DIR_RESULTADOS
import argparse
from tqdm import tqdm


def descargarConjuntoSiNecesario(nombreConjunto: str, configuracionConjunto: dict) -> str:
    """
    Descarga el dataset si no existe localmente
    
    Args:
        nombreConjunto: Nombre del dataset
        configuracionConjunto: Configuración del dataset
        
    Returns:
        Ruta del dataset
    """
    loader = CargadorKaggle()
    
    # Intentar descargar
    print(f"\nVerificando dataset: {nombreConjunto}")
    path = loader.descargarConjunto(configuracionConjunto['idKaggle'])
    
    if path:
        print(f"Dataset disponible en: {path}")
        return path
    else:
        raise RuntimeError(f"No se pudo descargar el dataset {nombreConjunto}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Procesador de imágenes multi-conjunto')
    parser.add_argument('--conjunto', choices=['ecommerce', 'mechanical_tools', 'todos'],
                       default='todos', help='Conjunto a procesar')
    parser.add_argument('--algoritmos', nargs='+', 
                       default=['momentos', 'sift', 'hog', 'embeddings'],
                       help='Algoritmos a aplicar')
    parser.add_argument('--muestra', type=int, default=None,
                       help='Número de imágenes a procesar (None = todas)')
    parser.add_argument('--salida', default=None,
                       help='Directorio de salida (default: results dentro de procesador_imagenes)')
    parser.add_argument('--correccion-automatica', action='store_true', default=None,
                       help='Activar análisis y corrección automática de imágenes (None usa config)')
    parser.add_argument('--fondo-blanco', action='store_true', default=True,
                       help='Las imágenes tienen fondo blanco (para análisis automático)')
    parser.add_argument('--guardar-imagenes', action='store_true', default=True,
                       help='Guardar imágenes procesadas en disco (default: True)')
    parser.add_argument('--directorio-imagenes', default=None,
                       help='Directorio donde guardar imágenes procesadas (default: {salida}/imagenes_procesadas)')
    parser.add_argument('--graficar-histograma', action='store_true', default=False,
                       help='Graficar histograma de las imágenes procesadas')
    
    args = parser.parse_args()
    
    # Resolver corrección automática con fallback a configuración
    correccion_automatica = args.correccion_automatica if args.correccion_automatica is not None else CONFIGURACION_PREPROCESAMIENTO.get('correccionAutomatica', False)
    dir_salida = args.salida or DIR_RESULTADOS

    # Limpiar carpeta de resultados si existe
    if os.path.exists(dir_salida):
        print(f"\nBorrando carpeta: {dir_salida}")
        shutil.rmtree(dir_salida)
        print(f"Carpeta borrada exitosamente\n")

    # Seleccionar datasets
    conjuntosAProcesar = []
    if args.conjunto == 'todos':
        conjuntosAProcesar = list(CONJUNTOS_DATOS.keys())
    else:
        conjuntosAProcesar = [args.conjunto]

    # Procesar cada dataset
    for nombreConjunto in conjuntosAProcesar:
        print(f"\n{'='*60}")
        print(f"Procesando dataset: {nombreConjunto}")
        print(f"{'='*60}\n")
        
        procesarConjunto(
            nombreConjunto, 
            args.algoritmos, 
            args.muestra, 
            dir_salida,
            correccion_automatica,
            args.fondo_blanco,
            args.guardar_imagenes,
            args.directorio_imagenes,
            args.graficar_histograma
        )


def procesarConjunto(nombreConjunto: str, algoritmos: list, 
                   tamanoMuestra: int = None, dirSalida: str = './results',
                   correccionAutomatica: bool = False, fondoBlanco: bool = True,
                   guardarImagenes: bool = False, dirImagenes: str = None,
                   graficarHistograma: bool = False):
    """
    Procesa un dataset específico
    
    Args:
        nombreConjunto: Nombre del dataset
        algoritmos: Lista de algoritmos
        tamanoMuestra: Número de imágenes a procesar
        dirSalida: Directorio de salida
        correccionAutomatica: Si aplicar corrección automática de imágenes
        fondoBlanco: Si las imágenes tienen fondo blanco
        guardarImagenes: Si guardar las imágenes procesadas
        dirImagenes: Directorio donde guardar imágenes procesadas
        graficarHistograma: Si graficar histogramas de las imágenes
    """
    
    configuracion = CONJUNTOS_DATOS[nombreConjunto]
    tamanoObjetivo = configuracion['tamanoObjetivo']
    
    print(f"Tamaño objetivo: {tamanoObjetivo}")
    print(f"Corrección automática: {'ACTIVADA' if correccionAutomatica else 'DESACTIVADA'}")
    if guardarImagenes:
        dirImagenesFinal = dirImagenes or os.path.join(dirSalida, nombreConjunto, 'imagenes_procesadas')
        print(f"Guardado de imágenes: ACTIVADO")
        print(f"Directorio de imágenes: {dirImagenesFinal}")
        # Crear directorio si no existe
        os.makedirs(dirImagenesFinal, exist_ok=True)
    
    # Descargar dataset automáticamente
    try:
        dataset_path = descargarConjuntoSiNecesario(nombreConjunto, configuracion)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Cargar dataset con filtros
    try:
        dataset_manager = GestorConjuntoDatos(
            dataset_path,
            categorias=configuracion.get('categorias'),
            subcarpeta=configuracion.get('subcarpeta')
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Obtener imágenes
    image_paths = dataset_manager.obtenerImagenes()
    

    if tamanoMuestra is not None:
        image_paths = image_paths[:tamanoMuestra]
    
    if len(image_paths) == 0:
        print("No se encontraron imágenes en el dataset")
        return
    
    # Mostrar estadísticas
    print(f"\nEstadísticas del dataset:")
    print(f"   Total de imágenes: {len(image_paths)}")
    
    try:
        stats = dataset_manager.obtenerEstadisticasImagenes()
        if stats:
            print(f"   Tamaño promedio: {stats.get('alturaPromedio', 0):.0f}x{stats.get('anchoPromedio', 0):.0f}")
            print(f"   Tamaño más común: {stats.get('tamanoMasComun', (0,0))}")
    except:
        pass
    
    # Procesar características
    print(f"\nExtrayendo características con algoritmos: {', '.join(algoritmos)}")
    
    processor = ProcesadorCaracteristicas(
        tamanoObjetivo=tamanoObjetivo
    )
    
    # Nuevo flujo: preprocesar una vez por imagen y extraer todo
    resultados_imagenes = []
    print(f"\nProcesando imágenes...")
    prep = PreprocesadorImagen(tamanoObjetivo)
    cfg = CONFIGURACION_PREPROCESAMIENTO
    contadorPorClase = {}  # Contador de imágenes por clase
    for i, ruta in enumerate(tqdm(image_paths, desc="Procesando", unit="img"), start=1):
        try:
            #1.- Cargar imagen
            imagenOriginal = prep.cargarImagen(ruta)

            #2.- Preprocesar imagen

            resultados= prep.preprocesarImagen(
                imagenOriginal,
                espacioColor=cfg.get('espacioColor', 'grayscale'),
                normalizar=cfg.get('normalizar', True),
                binarizar=True,
                correccionAutomatica=correccionAutomatica,
                fondoBlanco=fondoBlanco
            )
            procBase, analisisBase, binaria=(
                resultados.get('suavizada'),#Imagen preprocesada y eliminada de ruido
                resultados.get('analisisImagen'),#
                resultados.get('binarizada')
            )


            # 3.- Extraer características (Momentos, SIFT, HOG, embeddings)
            res = processor.extraerDesdePreprocesadas(
                ruta,
                imagenOriginal,  # Imagen original en color (RGB/BGR)
                procBase,  # Imagen preprocesada y eliminada de ruido (grayscale)
                binaria,   # Imagen binarizada
                algoritmos,
            )
            # Adjuntar análisis si existe
            if analisisBase and res.get('analisisImagen') is None:
                res['analisisImagen'] = analisisBase

            # Guardado de imágenes si se solicita
            if guardarImagenes:
                nombre = os.path.basename(ruta)
                # Obtener clase de la imagen (directorio padre)
                clase = Path(ruta).parent.name
                
                # Incrementar contador para esta clase
                if clase not in contadorPorClase:
                    contadorPorClase[clase] = 0
                contadorPorClase[clase] += 1
                
                # Estructura: clase/img{contador}/archivos
                carpeta_destino = os.path.join(dirImagenesFinal, clase, f"img{contadorPorClase[clase]}")
                rutas_destino = {
                    'original': os.path.join(carpeta_destino, f"original_{nombre}"),
                    'preprocesada': os.path.join(carpeta_destino, f"preprocesada_{nombre}"),
                    'binaria': os.path.join(carpeta_destino, f"binaria_{nombre}"),
                    'histograma': os.path.join(carpeta_destino, f"histograma_{nombre}")
                }
                imgs_a_guardar = {
                    'original': imagenOriginal,
                    'preprocesada': procBase,
                    'binaria': binaria
                }
                for clave, img in imgs_a_guardar.items():
                    if img is None:
                        continue
                    destino = rutas_destino[clave]
                    os.makedirs(os.path.dirname(destino), exist_ok=True)
                    prep.guardarImagen(img, destino, crearDirectorio=False)
                # Guardar histograma si está activado
                if graficarHistograma:
                    hist_destino = rutas_destino['histograma']
                    prep.graficarHistogramaImagen(resultados.get('gris'),
                                                  titulo=f"Imagen: {nombre}",
                                                  guardarRuta=hist_destino,
                                                  fondoBlanco=fondoBlanco,
                                                  analisis=analisisBase)


            resultados_imagenes.append(res)
        except Exception as e:
            resultados_imagenes.append({'rutaImagen': ruta, 'estado': 'error', 'error': str(e), 'caracteristicas': {}, 'analisisImagen': None})

    results = {
        'totalImagenes': len(image_paths),
        'algoritmos': algoritmos,
        'correccionAutomaticaForzada': correccionAutomatica,
        'fondoBlanco': fondoBlanco,
        'imagenes': resultados_imagenes,
        'estadisticas': processor._calcularEstadisticas({'imagenes': resultados_imagenes})
    }
    
    # Gestionar resultados
    results_handler = GestorResultados(os.path.join(dirSalida, nombreConjunto))
    
    # Guardar resultados
    print(f"\nGuardando resultados...")
    results_handler.guardarResultadosJson(results, 'caracteristicas.json')
    results_handler.guardarResultadosCsv(results, 'caracteristicas.csv')
    results_handler.guardarVectoresCaracteristicas(results)
    
    # Mostrar resumen
    print()
    results_handler.imprimirResumen(results)


if __name__ == '__main__':
    main()
