"""
Gestor de datasets para procesamiento
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from tqdm import tqdm


class GestorConjuntoDatos:
    """
    Gestiona datasets de imágenes para procesamiento
    """
    
    # Extensiones de imagen soportadas
    EXTENSIONES_SOPORTADAS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    def __init__(self, rutaConjunto: str, categorias: Optional[List[str]] = None, subcarpeta: Optional[str] = None):
        """
        Inicializa el gestor de dataset
        
        Args:
            rutaConjunto: Ruta al dataset
            categorias: Lista de categorías a incluir (None = todas)
            subcarpeta: Subcarpeta específica dentro del dataset
        """
        self.rutaConjunto = rutaConjunto
        self.categorias = categorias
        self.subcarpeta = subcarpeta
        self.archivosImagen = []
        self._descubrirImagenes()
    
    def _descubrirImagenes(self):
        """Descubre todas las imágenes en el dataset, aplicando filtros si existen"""
        ruta = Path(self.rutaConjunto)

        if not ruta.exists():
            raise ValueError(f"No existe la ruta del conjunto: {self.rutaConjunto}")
        
        # Si hay subcarpeta, ajustar la ruta
        if self.subcarpeta:
            rutaBusqueda = ruta / self.subcarpeta
            if not rutaBusqueda.exists():
                rutaBusqueda = ruta
        else:
            rutaBusqueda = ruta

        # Descubrir imagenes
        for ext in self.EXTENSIONES_SOPORTADAS:
            encontrados = list(rutaBusqueda.rglob(f'*{ext}'))
            encontrados.extend(rutaBusqueda.rglob(f'*{ext.upper()}'))

            # Filtrar por categorías si se especifican
            if self.categorias:
                filtrados = []
                for archivo in encontrados:
                    # Verificar si el archivo está en una de las categorías
                    partes = archivo.parts
                    if any(cat in partes for cat in self.categorias):
                        filtrados.append(archivo)
                self.archivosImagen.extend(filtrados)
            else:
                self.archivosImagen.extend(encontrados)
            
        # Remover duplicados
        self.archivosImagen = list(set(self.archivosImagen))
        self.archivosImagen.sort()

        print(f"Encontradas {len(self.archivosImagen)} imágenes en {self.rutaConjunto}")
        if self.categorias:
            print(f"  Categorías filtradas: {', '.join(self.categorias)}")
    
    def obtenerImagenes(self) -> List[str]:
        """
        Retorna lista de rutas de imágenes
        
        Returns:
            Lista de rutas de imágenes
        """
        return [str(img) for img in self.archivosImagen]
    
    def contarImagenes(self) -> int:
        """Retorna el número de imágenes"""
        return len(self.archivosImagen)
    
    def cargarImagen(self, rutaImagen: str) -> Optional[np.ndarray]:
        """
        Carga una imagen
        
        Args:
            rutaImagen: Ruta de la imagen
            
        Returns:
            Imagen cargada o None si hay error
        """
        try:
            imagen = cv2.imread(str(rutaImagen))
            return imagen
        except Exception as e:
            print(f"Error cargando imagen {rutaImagen}: {e}")
            return None
    
    def iterarLotes(self, tamanoLote: int = 32) -> Generator[List[str], None, None]:
        """
        Itera sobre el dataset en lotes
        
        Args:
            tamanoLote: Tamaño del lote
            
        Yields:
            Lotes de rutas de imágenes
        """
        for i in range(0, len(self.archivosImagen), tamanoLote):
            lote = [str(img) for img in self.archivosImagen[i:i+tamanoLote]]
            yield lote
    
    def obtenerEstadisticasImagenes(self) -> dict:
        """
        Calcula estadísticas del dataset
        
        Returns:
            Diccionario con estadísticas
        """
        if len(self.archivosImagen) == 0:
            return {}
        
        tamanos = []

        print("Calculando estadísticas del conjunto...")
        for ruta in tqdm(self.archivosImagen[:min(100, len(self.archivosImagen))]):
            img = self.cargarImagen(str(ruta))
            if img is not None:
                tamanos.append(img.shape)

        if len(tamanos) == 0:
            return {}
        
        alturas = [s[0] for s in tamanos]
        anchos = [s[1] for s in tamanos]

        return {
            'totalImagenes': len(self.archivosImagen),
            'muestra': len(tamanos),
            'alturaPromedio': float(np.mean(alturas)),
            'anchoPromedio': float(np.mean(anchos)),
            'alturaMin': int(np.min(alturas)),
            'alturaMax': int(np.max(alturas)),
            'anchoMin': int(np.min(anchos)),
            'anchoMax': int(np.max(anchos)),
            'tamanoMasComun': self._tamanoMasComun(tamanos)
        }
    
    def _tamanoMasComun(self, tamanos: List[Tuple]) -> Tuple[int, int]:
        """Obtiene el tamaño más común"""
        from collections import Counter
        conteos = Counter([(s[0], s[1]) for s in tamanos])
        masComun = conteos.most_common(1)
        return masComun[0][0] if masComun else (0, 0)
    
    def filtrarPorTamano(self, minimo: Tuple[int, int] = (100, 100),
                         maximo: Tuple[int, int] = (4096, 4096)) -> List[str]:
        """
        Filtra imágenes por tamaño
        
        Args:
            minimo: Tamaño mínimo (altura, ancho)
            maximo: Tamaño máximo (altura, ancho)
            
        Returns:
            Lista de rutas que cumplen criterio de tamaño
        """
        validas = []

        print("Filtrando imágenes por tamaño...")
        for ruta in tqdm(self.archivosImagen):
            img = self.cargarImagen(str(ruta))
            if img is not None:
                h, w = img.shape[:2]
                if (minimo[0] <= h <= maximo[0] and 
                    minimo[1] <= w <= maximo[1]):
                    validas.append(str(ruta))

        print(f"Imágenes válidas: {len(validas)}/{len(self.archivosImagen)}")
        return validas
