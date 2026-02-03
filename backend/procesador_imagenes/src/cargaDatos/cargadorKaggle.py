"""
Gestor de carga de datos de Kaggle usando kagglehub
"""

import os
from typing import Optional, List
import kagglehub


class CargadorKaggle:
    """
    Carga datasets desde Kaggle usando kagglehub
    """
    
    def __init__(self):
        """
        Inicializa el cargador de datos de Kaggle
        """
        pass
    
    def descargarConjunto(self, idKaggle: str) -> Optional[str]:
        """
        Descarga un dataset de Kaggle usando kagglehub
        
        Args:
            idKaggle: ID del dataset (formato: user/dataset)
            
        Returns:
            Ruta del dataset descargado o None si falla
        """
        try:
            print(f"Descargando dataset: {idKaggle}")
            ruta = kagglehub.dataset_download(idKaggle)
            print(f"Dataset descargado en: {ruta}")
            return ruta
        except Exception as e:
            print(f"Error descargando dataset: {e}")
            return None
    
    def listarArchivosConjunto(self, rutaConjunto: str) -> List[str]:
        """
        Lista los archivos en un dataset
        
        Args:
            rutaConjunto: Ruta del dataset
            
        Returns:
            Lista de archivos encontrados
        """
        archivos = []
        if os.path.exists(rutaConjunto):
            for raiz, directorios, nombres in os.walk(rutaConjunto):
                for nombre in nombres:
                    archivos.append(os.path.join(raiz, nombre))
        return archivos
