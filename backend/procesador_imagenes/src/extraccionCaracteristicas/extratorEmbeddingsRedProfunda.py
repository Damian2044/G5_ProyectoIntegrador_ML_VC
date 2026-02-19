"""Extractor de embeddings usando una red profunda ResNet18.

Esta clase carga (o descarga si no existe) un modelo ResNet18
pre-entrenado y lo utiliza como extractor de características.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

import torch
from torchvision import models, transforms


class ExtractorEmbeddingsRedProfunda:
    """Extractor de embeddings basado en ResNet18.

    Flujo de uso típico:

    1. Se instancia la clase, que comprueba si el archivo de pesos
       existe en la carpeta actual; si no existe, descarga ResNet18
       pre-entrenada en ImageNet y guarda los pesos.
    2. Se llama a ``extraerEmbeddings`` pasando una imagen en formato
       NumPy (BGR o RGB). La imagen se visualiza y se devuelve el
       vector de embeddings de la penúltima capa.
    """

    def __init__(
        self,
        rutaModelo: str | Path = "resnet18_imagenet.pth",
        usarGpuSiDisponible: bool = True,
    ) -> None:
        """Inicializa el extractor y prepara el modelo ResNet18.

        Args:
            rutaModelo: Nombre o ruta del archivo de pesos a utilizar.
                Si el archivo no existe en la carpeta de trabajo, se
                descarga ResNet18 pre-entrenada y se guardan sus pesos
                en esa ruta.
            usarGpuSiDisponible: Si es True, usa CUDA cuando esté
                disponible; en caso contrario fuerza CPU.
        """

        # Guardar siempre el modelo en la carpeta de este archivo,
        # no en el directorio desde el que se ejecuta el script.
        rutaBaseArchivo = Path(__file__).resolve().parent
        self.rutaModelo = rutaBaseArchivo / Path(rutaModelo)
        if usarGpuSiDisponible and torch.cuda.is_available():
            self.dispositivo = torch.device("cuda")
        else:
            self.dispositivo = torch.device("cpu")

        self.modelo = self._cargarModeloResnet18()
        self.modelo.eval()

        # Transformación estándar para ResNet18 (ImageNet)
        self.transformacionImagen = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _cargarModeloResnet18(self) -> torch.nn.Module:
        """Carga ResNet18 como extractor de características.

        - Si ``rutaModelo`` existe, se cargan los pesos desde disco.
        - Si no existe, se descarga ResNet18 pre-entrenada en ImageNet,
          se guarda ``state_dict`` en ``rutaModelo`` y se reutiliza a
          partir de entonces.
        """
        print("Cargando ResNet18...")

        from torchvision.models import ResNet18_Weights

        if self.rutaModelo.exists():
            modelo = models.resnet18(weights=None)
            estado = torch.load(self.rutaModelo, map_location="cpu")
            modelo.load_state_dict(estado, strict=True)
        else:
            pesos = ResNet18_Weights.IMAGENET1K_V1
            modelo = models.resnet18(weights=pesos)
            torch.save(modelo.state_dict(), self.rutaModelo)


        # Usar la salida de la penúltima capa como embedding
        modelo.fc = torch.nn.Identity()
        modelo.to(self.dispositivo)
        return modelo

    def _preprocesarImagen(self, imagen: np.ndarray) -> torch.Tensor:
        """Preprocesa una imagen NumPy para pasarla por ResNet18.

        La imagen de entrada se espera normalmente en formato BGR
        (como la devuelve OpenCV). Aquí se convierte a RGB, se adapta
        a 3 canales en caso de ser en escala de grises y se aplican
        las transformaciones estándar de ImageNet.

        Args:
            imagen: Matriz NumPy con forma (alto, ancho, 3) en BGR o
                (alto, ancho) en escala de grises.

        Returns:
            Tensor de PyTorch con forma (1, 3, 224, 224) listo para
            ser pasado por el modelo.
        """

        if imagen.ndim == 2:
            imagenRgb = np.stack([imagen, imagen, imagen], axis=-1)
        elif imagen.ndim == 3 and imagen.shape[2] == 3:
            # Se asume BGR y se convierte a RGB invirtiendo canales
            imagenRgb = imagen[..., ::-1]
        else:
            raise ValueError(
                "La imagen debe tener forma (H, W) o (H, W, 3).",
            )

        if imagenRgb.dtype != np.uint8:
            imagenRgb = imagenRgb.astype(np.uint8)

        tensor = self.transformacionImagen(imagenRgb)
        tensor = tensor.unsqueeze(0).to(self.dispositivo)
        return tensor

    def extraerEmbeddings(self, imagen: np.ndarray) -> np.ndarray:
        """Extrae el vector de embedding de una imagen.

        Args:
            imagen: Imagen como matriz NumPy. Habitualmente será la
                salida de ``cv2.imread``, es decir, en formato BGR
                con forma (alto, ancho, 3). También se acepta una
                imagen en escala de grises con forma (alto, ancho).

        Returns:
            Un vector NumPy 1D (float32) con el embedding generado por
            la penúltima capa de ResNet18.
        """

        with torch.no_grad():
            tensorEntrada = self._preprocesarImagen(imagen)
            salida = self.modelo(tensorEntrada)

        embedding = salida.cpu().numpy().reshape(-1).astype(np.float32)
        return embedding




