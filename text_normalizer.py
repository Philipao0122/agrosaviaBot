# -*- coding: utf-8 -*-
"""
Módulo para normalización de texto en español.
Maneja tildes, variantes ortográficas y sinónimos comunes.
"""
import re
import unicodedata
from typing import Dict, List, Set, Optional

# Diccionario de reemplazos para normalización de palabras
REPLACEMENTS = {
    # Tildes y caracteres especiales
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    'ü': 'u', 'ñ': 'n',
    # Variantes comunes - se han eliminado reemplazos demasiado agresivos
    # 'y': 'i',  # Eliminado para evitar problemas con palabras como 'yuca'
    'z': 's',  # En algunos casos como 'zapallo' vs 'sapallo'
    # 'b': 'v',  # Comentado para evitar problemas con palabras como 'vaca' vs 'baca'
    # 'll': 'y', # Comentado para evitar problemas con palabras como 'lluvia' vs 'yuvia'
    'h': '',   # H muda - mantener este reemplazo
    
    # Variantes de palabras específicas comunes en agricultura
    'platano': 'plátano',
    'frijol': 'fríjol',
    'maiz': 'maíz',
    'papa': 'papa',  # Mantener como está
    'patata': 'papa',  # Sinónimo
    'cana': 'caña',
    'guayaba': 'guayabo',
    'cafe': 'café',
    'citrico': 'cítrico',
    'citricos': 'cítricos',
    'citrica': 'cítrica',
    'citricas': 'cítricas',
    'yuca': 'yuca',  # Asegurar que 'yuca' no se modifique
    'iuca': 'yuca',  # Mapear 'iuca' a 'yuca' por si acaso
}

# Expresiones regulares para limpieza de texto
MULTIPLE_SPACES = re.compile(r'\s+')
NON_WORD_CHARS = re.compile(r'[^\w\s]', re.UNICODE)

class TextNormalizer:
    """Clase para normalizar texto en español."""
    
    def __init__(self, custom_replacements: Optional[Dict[str, str]] = None):
        """
        Inicializa el normalizador con reemplazos personalizados opcionales.
        
        Args:
            custom_replacements: Diccionario con reemplazos personalizados que se añadirán
                               a los reemplazos por defecto.
        """
        self.replacements = REPLACEMENTS.copy()
        if custom_replacements:
            self.replacements.update(custom_replacements)
            
        # Compilar expresión regular para reemplazos más rápidos
        self.regex_replacements = re.compile("|".join(
            re.escape(key) for key in self.replacements.keys()
        ))
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliza un texto eliminando tildes, caracteres especiales y estandarizando variantes.
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Convertir a minúsculas
        text = text.lower()
        
        # Reemplazar caracteres especiales usando el diccionario
        def replace_match(match):
            return self.replacements.get(match.group(0), match.group(0))
            
        text = self.regex_replacements.sub(replace_match, text)
        
        # Eliminar tildes usando normalización Unicode NFD
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'  # Quitar marcas diacríticas
        )
        
        # Eliminar caracteres no alfanuméricos (excepto espacios)
        text = NON_WORD_CHARS.sub('', text)
        
        # Reemplazar múltiples espacios por uno solo
        text = MULTIPLE_SPACES.sub(' ', text).strip()
        
        return text
    
    def normalize_list(self, texts: List[str]) -> List[str]:
        """
        Normaliza una lista de textos.
        
        Args:
            texts: Lista de textos a normalizar
            
        Returns:
            Lista de textos normalizados
        """
        return [self.normalize_text(text) for text in texts]
    
    def get_synonyms(self, word: str) -> Set[str]:
        """
        Obtiene sinónimos de una palabra basado en los reemplazos definidos.
        
        Args:
            word: Palabra para la cual buscar sinónimos
            
        Returns:
            Conjunto de sinónimos incluyendo la palabra original
        """
        word = self.normalize_text(word)
        synonyms = {word}
        
        # Buscar la palabra como valor en los reemplazos
        for key, value in self.replacements.items():
            if value == word:
                synonyms.add(key)
            elif key == word:
                synonyms.add(value)
                
        return synonyms

# Instancia global para uso directo
normalizer = TextNormalizer()

def normalize_text(text: str) -> str:
    """Función de conveniencia para normalizar texto."""
    return normalizer.normalize_text(text)

def normalize_list(texts: List[str]) -> List[str]:
    """Función de conveniencia para normalizar una lista de textos."""
    return normalizer.normalize_list(texts)

def get_synonyms(word: str) -> Set[str]:
    """Función de conveniencia para obtener sinónimos de una palabra."""
    return normalizer.get_synonyms(word)
