"""
Módulo de extracción de características
"""

from .extractorMomentos import ExtractorMomentos
from .extractorSift import ExtractorSift
from .extractorHog import ExtractorHog
from .extratorEmbeddingsRedProfunda import ExtractorEmbeddingsRedProfunda

__all__ = [
	'ExtractorMomentos',
	'ExtractorSift',
	'ExtractorHog',
	'ExtractorEmbeddingsRedProfunda',
]
