# -*- coding: utf-8 -*-
"""
Módulo que contiene información detallada sobre cultivos agrícolas.
Incluye variantes ortográficas, nombres científicos y sinónimos comunes.
"""
from typing import Dict, List, Set

# Diccionario principal de cultivos con sus variantes
CULTIVOS: Dict[str, Dict[str, List[str]]] = {
    'plátano': {
        'variantes': ['platano', 'banano', 'banana', 'cambur', 'guineo', 'topocho'],
        'cientifico': ['Musa × paradisiaca'],
        'sinonimos': ['bananero', 'platanero']
    },
    'yuca': {
        'variantes': ['mandioca', 'guacamota', 'casabe', 'aipim', 'macaxeira'],
        'cientifico': ['Manihot esculenta'],
        'sinonimos': ['manihot', 'yuca amarga', 'yuca dulce']
    },
    'arroz': {
        'variantes': ['arros', 'arroç', 'aroz'],
        'cientifico': ['Oryza sativa'],
        'sinonimos': []
    },
    'maíz': {
        'variantes': ['maiz', 'mais', 'elote', 'choclo', 'jojoto', 'millo'],
        'cientifico': ['Zea mays'],
        'sinonimos': ['maíz dulce', 'maíz tierno']
    },
    'café': {
        'variantes': ['cafe', 'cafeto'],
        'cientifico': ['Coffea arabica', 'Coffea canephora'],
        'sinonimos': ['café arábigo', 'café robusta']
    },
    'frijol': {
        'variantes': ['fríjol', 'poroto', 'caraota', 'alubia', 'judía', 'habichuela'],
        'cientifico': ['Phaseolus vulgaris'],
        'sinonimos': ['fréjol', 'poroto']
    },
    'papa': {
        'variantes': ['patata', 'papa andina', 'papa criolla'],
        'cientifico': ['Solanum tuberosum'],
        'sinonimos': ['patata', 'papa común']
    },
    'caña de azúcar': {
        'variantes': ['caña', 'caña dulce', 'cañaveral'],
        'cientifico': ['Saccharum officinarum'],
        'sinonimos': ['cañaduz', 'cañamelar']
    },
    'papa criolla': {
        'variantes': ['papa amarilla', 'papa sabanera', 'criolla'],
        'cientifico': ['Solanum phureja'],
        'sinonimos': ['papa criolla colombiana']
    },
    'aguacate': {
        'variantes': ['palta', 'avocado'],
        'cientifico': ['Persea americana'],
        'sinonimos': ['aguacatero', 'palto']
    },
    'cacao': {
        'variantes': ['cocoa'],
        'cientifico': ['Theobroma cacao'],
        'sinonimos': ['árbol del cacao']
    },
    'palma africana': {
        'variantes': ['palma de aceite', 'palmiche'],
        'cientifico': ['Elaeis guineensis'],
        'sinonimos': ['palmera aceitera']
    },
    'arracacha': {
        'variantes': ['apio criollo', 'zanahoria blanca', 'virraca'],
        'cientifico': ['Arracacia xanthorrhiza'],
        'sinonimos': ['racacha', 'zanahoria blanca']
    },
    'guayaba': {
        'variantes': ['guayabo', 'guayabero'],
        'cientifico': ['Psidium guajava'],
        'sinonimos': ['guayabero']
    },
    'mango': {
        'variantes': ['mangó'],
        'cientifico': ['Mangifera indica'],
        'sinonimos': ['manguero']
    },
    'piña': {
        'variantes': ['ananá', 'ananás'],
        'cientifico': ['Ananas comosus'],
        'sinonimos': ['ananá', 'piña americana']
    },
    'tomate': {
        'variantes': ['jitomate', 'tomatera'],
        'cientifico': ['Solanum lycopersicum'],
        'sinonimos': ['tomate rojo']
    },
    'cebolla': {
        'variantes': ['cebolla blanca', 'cebolla morada', 'cebollín'],
        'cientifico': ['Allium cepa'],
        'sinonimos': ['cebolla común']
    },
    'zanahoria': {
        'variantes': ['zanahoria naranja', 'zanahoria morada'],
        'cientifico': ['Daucus carota'],
        'sinonimos': ['zanahoria común']
    },
    'lechuga': {
        'variantes': ['lechuga crespa', 'lechuga romana', 'lechuga mantecosa'],
        'cientifico': ['Lactuca sativa'],
        'sinonimos': []
    }
}

def obtener_todas_variantes() -> Dict[str, List[str]]:
    """
    Devuelve un diccionario con todas las variantes de los cultivos.
    
    Returns:
        Dict[str, List[str]]: Diccionario donde las claves son los nombres principales
        y los valores son listas con todas las variantes, incluyendo nombres científicos y sinónimos.
    """
    resultado = {}
    for nombre, datos in CULTIVOS.items():
        todas_variantes = [nombre.lower()]
        todas_variantes.extend(v.lower() for v in datos['variantes'])
        todas_variantes.extend(v.lower() for v in datos['cientifico'])
        todas_variantes.extend(v.lower() for v in datos['sinonimos'])
        
        # Eliminar duplicados manteniendo el orden
        unicas = []
        visto = set()
        for v in todas_variantes:
            if v not in visto:
                visto.add(v)
                unicas.append(v)
                
        resultado[nombre] = unicas
    
    return resultado

def buscar_cultivo_por_nombre(nombre: str) -> Dict[str, List[str]]:
    """
    Busca un cultivo por cualquiera de sus nombres o variantes.
    
    Args:
        nombre: Nombre o variante del cultivo a buscar
        
    Returns:
        Dict[str, List[str]]: Diccionario con la información del cultivo si se encuentra,
        o None si no se encuentra.
    """
    nombre = nombre.lower().strip()
    
    for cultivo, datos in CULTIVOS.items():
        # Verificar contra el nombre principal
        if nombre == cultivo.lower():
            return {cultivo: datos}
            
        # Verificar contra variantes
        if nombre in (v.lower() for v in datos['variantes']):
            return {cultivo: datos}
            
        # Verificar contra nombres científicos
        if nombre in (v.lower() for v in datos['cientifico']):
            return {cultivo: datos}
            
        # Verificar contra sinónimos
        if nombre in (v.lower() for v in datos['sinonimos']):
            return {cultivo: datos}
    
    return {}

def obtener_todos_cultivos() -> List[str]:
    """
    Devuelve una lista con los nombres principales de todos los cultivos.
    
    Returns:
        List[str]: Lista de nombres de cultivos.
    """
    return sorted(CULTIVOS.keys())

if __name__ == "__main__":
    # Ejemplos de uso
    print("Cultivos disponibles:")
    for i, cultivo in enumerate(obtener_todos_cultivos(), 1):
        print(f"{i}. {cultivo}")
    
    # Ejemplo de búsqueda
    print("\nBuscando 'banana':")
    print(buscar_cultivo_por_nombre("banana"))
    
    print("\nTodas las variantes de 'yuca':")
    print(obtener_todas_variantes()['yuca'])
