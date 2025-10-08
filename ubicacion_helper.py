# -*- coding: utf-8 -*-
"""
Módulo auxiliar para manejar la lógica de ubicación de cultivos.
"""
from typing import Dict, List, Optional, Tuple
import re
from text_normalizer import TextNormalizer
from cultivos import CULTIVOS, buscar_cultivo_por_nombre, obtener_todas_variantes

class UbicacionHelper:
    """Clase auxiliar para manejar la lógica de ubicación de cultivos."""
    
    def __init__(self):
        """Inicializa el helper con un normalizador de texto."""
        self.normalizer = TextNormalizer()
        self.cultivo_objetivo = None
        self.variantes_cultivo = set()
        
    def extraer_cultivo(self, pregunta: str) -> Optional[str]:
        """
        Extrae el nombre del cultivo de una pregunta sobre ubicación.
        
        Args:
            pregunta: Texto de la pregunta del usuario
            
        Returns:
            str: Nombre del cultivo normalizado si se encuentra, None en caso contrario
        """
        if not pregunta:
            return None
            
        # Normalizar la pregunta
        pregunta = self.normalizer.normalize_text(pregunta)
        
        # Buscar coincidencias directas con los cultivos conocidos
        for cultivo, datos in CULTIVOS.items():
            # Comprobar el nombre principal
            if self._coincidencia_exacta(cultivo, pregunta):
                self._configurar_variantes(cultivo)
                return cultivo
                
            # Comprobar variantes
            for variante in datos['variantes']:
                if self._coincidencia_exacta(variante, pregunta):
                    self._configurar_variantes(cultivo)
                    return cultivo
                    
            # Comprobar nombres científicos
            for cientifico in datos['cientifico']:
                if self._coincidencia_exacta(cientifico, pregunta):
                    self._configurar_variantes(cultivo)
                    return cultivo
                    
            # Comprobar sinónimos
            for sinonimo in datos['sinonimos']:
                if self._coincidencia_exacta(sinonimo, pregunta):
                    self._configurar_variantes(cultivo)
                    return cultivo
        
        return None
    
    def _coincidencia_exacta(self, palabra: str, texto: str) -> bool:
        """Verifica si una palabra aparece como palabra completa en el texto."""
        # Usar regex para coincidencia de palabra completa
        return bool(re.search(rf'\b{re.escape(palabra.lower())}\b', texto.lower()))
    
    def _configurar_variantes(self, cultivo_principal: str):
        """Configura las variantes del cultivo objetivo."""
        self.cultivo_objetivo = cultivo_principal
        datos = CULTIVOS.get(cultivo_principal, {})
        
        # Crear un conjunto con todas las variantes del cultivo
        self.variantes_cultivo = set()
        self.variantes_cultivo.add(cultivo_principal.lower())
        self.variantes_cultivo.update(v.lower() for v in datos.get('variantes', []))
        self.variantes_cultivo.update(v.lower() for v in datos.get('cientifico', []))
        self.variantes_cultivo.update(v.lower() for v in datos.get('sinonimos', []))
    
    def validar_respuesta_ubicacion(self, respuesta: str) -> bool:
        """
        Valida si una respuesta de ubicación es relevante para el cultivo objetivo.
        
        Args:
            respuesta: Texto de la respuesta a validar
            
        Returns:
            bool: True si la respuesta es relevante, False en caso contrario
        """
        if not self.cultivo_objetivo or not self.variantes_cultivo:
            return True  # Si no hay cultivo objetivo, se acepta cualquier respuesta
            
        # Normalizar la respuesta
        respuesta = self.normalizer.normalize_text(respuesta)
        
        # Buscar coincidencias con cualquiera de las variantes del cultivo
        for variante in self.variantes_cultivo:
            if self._coincidencia_exacta(variante, respuesta):
                return True
                
        return False
    
    def obtener_patron_busqueda(self) -> str:
        """
        Devuelve un patrón de expresión regular para buscar el cultivo en textos.
        
        Returns:
            str: Patrón de expresión regular
        """
        if not self.variantes_cultivo:
            return r'\b\w+\b'  # Patrón genérico si no hay variantes
            
        # Crear un patrón que coincida con cualquiera de las variantes
        variantes_esc = [re.escape(v) for v in self.variantes_cultivo]
        return r'\b(' + '|'.join(variantes_esc) + r')\b'

# Instancia global para uso directo
helper_ubicacion = UbicacionHelper()

def extraer_cultivo_desde_pregunta(pregunta: str) -> Optional[str]:
    """
    Función de conveniencia para extraer el cultivo de una pregunta.
    
    Args:
        pregunta: Texto de la pregunta del usuario
        
    Returns:
        str: Nombre del cultivo si se encuentra, None en caso contrario
    """
    return helper_ubicacion.extraer_cultivo(pregunta)

def validar_respuesta_ubicacion(respuesta: str) -> bool:
    """
    Función de conveniencia para validar una respuesta de ubicación.
    
    Args:
        respuesta: Texto de la respuesta a validar
        
    Returns:
        bool: True si la respuesta es relevante, False en caso contrario
    """
    return helper_ubicacion.validar_respuesta_ubicacion(respuesta)

def obtener_patron_busqueda_cultivo() -> str:
    """
    Función de conveniencia para obtener el patrón de búsqueda del cultivo.
    
    Returns:
        str: Patrón de expresión regular para buscar el cultivo
    """
    return helper_ubicacion.obtener_patron_busqueda()
