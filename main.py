from fastapi import FastAPI, HTTPException, Query, Body
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
import random

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="API de Agrosavia",
    description="API para consultar y gestionar preguntas frecuentes de Agrosavia",
    version="1.0.0"
)

# Configuración de Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Las variables de entorno SUPABASE_URL y SUPABASE_ANON_KEY son requeridas.")

# Inicializar cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Modelos Pydantic
class QAItem(BaseModel):
    categoria: str
    pregunta: str
    respuesta: str
    respuesta_compuesta: Optional[str] = None

class QAUpdate(BaseModel):
    categoria: Optional[str] = None
    pregunta: Optional[str] = None
    respuesta: Optional[str] = None
    respuesta_compuesta: Optional[str] = None

# Endpoints de consulta
@app.get("/", tags=["Inicio"])
async def read_root():
    """Página de inicio de la API"""
    return {
        "message": "¡Bienvenido a la API de Agrosavia!",
        "endpoints": {
            "documentación": "/docs",
            "todas_preguntas": "/api/qa",
            "por_categoria": "/api/qa/categoria/{categoria}",
            "buscar": "/api/qa/buscar?q=texto",
            "aleatoria": "/api/qa/aleatoria",
            "estadisticas": "/api/estadisticas"
        }
    }

@app.get("/api/qa", tags=["Consultas"])
async def get_all_qa(limit: int = 10, offset: int = 0):
    """Obtiene todas las preguntas con paginación"""
    try:
        response = supabase.table('chatbot_qa')\
                         .select('*')\
                         .range(offset, offset + limit - 1)\
                         .execute()
        return {
            "data": response.data,
            "count": len(response.data),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/qa/categoria/{categoria}", tags=["Consultas"])
async def get_qa_by_category(categoria: str, limit: int = 10, offset: int = 0):
    """Busca preguntas por categoría"""
    try:
        response = supabase.table('chatbot_qa')\
                         .select('*')\
                         .eq('categoria', categoria)\
                         .range(offset, offset + limit - 1)\
                         .execute()
        return {
            "data": response.data,
            "categoria": categoria,
            "count": len(response.data),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/qa/buscar", tags=["Consultas"])
async def search_qa(q: str = Query(..., min_length=3), limit: int = 10):
    """Busca preguntas que contengan el texto en pregunta o respuesta"""
    try:
        response = supabase.table('chatbot_qa')\
                         .select('*')\
                         .ilike('pregunta', f'%{q}%')\
                         .limit(limit)\
                         .execute()
        
        # Buscar también en respuestas si no hay suficientes resultados
        if len(response.data) < limit:
            response_respuesta = supabase.table('chatbot_qa')\
                                     .select('*')\
                                     .ilike('respuesta', f'%{q}%')\
                                     .limit(limit - len(response.data))\
                                     .execute()
            # Combinar resultados únicos
            ids = {item['id'] for item in response.data}
            for item in response_respuesta.data:
                if item['id'] not in ids:
                    response.data.append(item)
                    ids.add(item['id'])
        
        return {
            "query": q,
            "results": response.data,
            "count": len(response.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/qa/aleatoria", tags=["Consultas"])
async def get_random_qa():
    """Obtiene una pregunta aleatoria"""
    try:
        # Obtener el conteo total
        count_response = supabase.table('chatbot_qa')\
                              .select('id', count='exact')\
                              .execute()
        
        if not count_response.count:
            raise HTTPException(status_code=404, detail="No se encontraron preguntas")
            
        # Obtener una pregunta aleatoria
        random_offset = random.randint(0, count_response.count - 1)
        response = supabase.table('chatbot_qa')\
                         .select('*')\
                         .range(random_offset, random_offset)\
                         .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="No se encontró la pregunta aleatoria")
            
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de gestión
@app.post("/api/qa", status_code=201, tags=["Gestión"])
async def create_qa(item: QAItem):
    """Agrega una nueva pregunta/respuesta"""
    try:
        response = supabase.table('chatbot_qa').insert(item.dict()).execute()
        if not response.data:
            raise HTTPException(status_code=400, detail="No se pudo crear el registro")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/qa/{item_id}", tags=["Gestión"])
async def update_qa(item_id: int, item: QAUpdate):
    """Actualiza una pregunta/respuesta existente"""
    try:
        update_data = {k: v for k, v in item.dict().items() if v is not None}
        if not update_data:
            raise HTTPException(status_code=400, detail="No se proporcionaron datos para actualizar")
            
        response = supabase.table('chatbot_qa')\
                         .update(update_data)\
                         .eq('id', item_id)\
                         .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"No se encontró el registro con ID {item_id}")
            
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/qa/{item_id}", status_code=204, tags=["Gestión"])
async def delete_qa(item_id: int):
    """Elimina una pregunta/respuesta"""
    try:
        response = supabase.table('chatbot_qa')\
                         .delete()\
                         .eq('id', item_id)\
                         .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"No se encontró el registro con ID {item_id}")
            
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuración de paginación para estadísticas
PAGE_SIZE = 1000  # Número de registros por página

# Cache simple para estadísticas (vaciar con POST /api/estadisticas/actualizar)
_stats_cache = None

# Función auxiliar para obtener datos paginados
def fetch_paginated_data(table, columns='*', page_size=1000, filter_cond=None):
    offset = 0
    all_data = []
    
    while True:
        query = supabase.table(table).select(columns).range(offset, offset + page_size - 1)
        
        if filter_cond:
            query = query.filter(**filter_cond)
            
        response = query.execute()
        
        if not hasattr(response, 'data') or not response.data:
            break
            
        all_data.extend(response.data)
        
        # Si recibimos menos registros que el tamaño de la página, es la última página
        if len(response.data) < page_size:
            break
            
        offset += page_size
    
    return all_data

# Endpoint para forzar la actualización de estadísticas
@app.post("/api/estadisticas/actualizar", tags=["Estadísticas"])
async def actualizar_estadisticas():
    """Fuerza la actualización de las estadísticas en caché"""
    global _stats_cache
    _stats_cache = None
    return {"status": "success", "message": "Caché de estadísticas borrado. La próxima consulta recalculará los datos."}

# Estadísticas optimizadas
@app.get("/api/estadisticas", tags=["Estadísticas"])
async def get_stats():
    """Obtiene estadísticas detalladas de la base de datos"""
    global _stats_cache
    
    # Usar caché si está disponible
    if _stats_cache:
        return {
            "status": "success",
            "estadisticas": _stats_cache,
            "nota": "Datos en caché. Use POST /api/estadisticas/actualizar para forzar actualización.",
            "cache": True
        }
    
    try:
        import time
        start_time = time.time()
        
        # 1. Verificar conexión con Supabase
        try:
            test_query = supabase.table('chatbot_qa').select('id').limit(1).execute()
            if hasattr(test_query, 'error') and test_query.error:
                raise Exception(f"Error de permisos: {test_query.error}")
        except Exception as e:
            return {
                "status": "error",
                "message": "Error al conectar con la base de datos",
                "details": str(e)
            }
        
        # 2. Obtener total de filas (usando count exacto)
        try:
            total_resp = supabase.table('chatbot_qa').select('id', count='exact').execute()
            total = total_resp.count if hasattr(total_resp, 'count') else 0
        except Exception as e:
            total = 0
        
        # 3. Filas con categoría NULL o vacía (optimizado)
        try:
            count_null_or_empty = 0
            # Primero contamos nulos
            null_count = supabase.table('chatbot_qa')\
                               .select('id', count='exact')\
                               .is_('categoria', 'null')\
                               .execute()
            count_null_or_empty += null_count.count if hasattr(null_count, 'count') else 0
            
            # Luego contamos vacíos
            empty_count = supabase.table('chatbot_qa')\
                               .select('id', count='exact')\
                               .eq('categoria', '')\
                               .execute()
            count_null_or_empty += empty_count.count if hasattr(empty_count, 'count') else 0
        except Exception as e:
            count_null_or_empty = 0
        
        # 4. Conteo por categoría (optimizado con paginación)
        categorias_exactas = {}
        categorias_normalizadas = {}
        separadores = {';': 0, '/': 0, '|': 0}
        
        try:
            # Obtenemos los datos paginados
            all_records = fetch_paginated_data('chatbot_qa', columns='categoria')
            
            # Procesamos los registros en lotes
            for item in all_records:
                cat = item.get('categoria', 'Sin categoría') or 'Sin categoría'
                
                # Conteo exacto
                categorias_exactas[cat] = categorias_exactas.get(cat, 0) + 1
                
                # Conteo normalizado (minúsculas y sin espacios)
                if cat and cat != 'Sin categoría':
                    cat_norm = cat.lower().strip()
                    if cat_norm:
                        categorias_normalizadas[cat_norm] = categorias_normalizadas.get(cat_norm, 0) + 1
                
                # Detección de separadores
                if isinstance(cat, str):
                    for sep in separadores.keys():
                        if sep in cat:
                            separadores[sep] += 1
            
            # Ordenar resultados
            categorias_exactas = dict(sorted(categorias_exactas.items(), key=lambda x: x[1], reverse=True))
            categorias_normalizadas = dict(sorted(categorias_normalizadas.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            if not categorias_exactas:
                categorias_exactas = {"error": f"Error al obtener categorías: {str(e)}"}
            if not categorias_normalizadas:
                categorias_normalizadas = {"error": f"Error al normalizar categorías: {str(e)}"}
        
        # Formatear resultado final
        estadisticas = {
            "total_registros": total,
            "categorias": {
                "sin_categoria": count_null_or_empty,
                "porcentaje_sin_categoria": round((count_null_or_empty / total * 100), 4) if total > 0 else 0,
                "conteo_por_categoria": categorias_exactas,
                "total_categorias_unicas": len(categorias_exactas)
            },
            "analisis_separadores": {
                "separador_punto_y_coma": separadores[';'],
                "separador_barra": separadores['/'],
                "separador_pipe": separadores['|']
            },
            "categorias_normalizadas": {
                "conteo": categorias_normalizadas,
                "total_categorias_unicas": len(categorias_normalizadas)
            },
            "metadatos": {
                "fecha_consulta": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version_api": "1.1.0",
                "tiempo_ejecucion_segundos": round(time.time() - start_time, 2)
            }
        }
        
        # Actualizar caché
        _stats_cache = estadisticas
        
        return {
            "status": "success",
            "estadisticas": estadisticas,
            "nota": "Datos calculados en tiempo real. Use POST /api/estadisticas/actualizar para forzar actualización.",
            "cache": False
        }
        
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(
            status_code=500, 
            detail={"message": "Error al calcular estadísticas", "details": error_details}
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )

# Este endpoint está duplicado y ya existe una versión mejorada en /api/qa/categoria/{categoria}
# Se recomienda usar el nuevo endpoint en su lugar
@app.get("/qa/category/{category_name}")
async def get_qa_by_category_old(category_name: str):
    """(Obsoleto) Usar /api/qa/categoria/{categoria} en su lugar"""
    try:
        response = supabase.table('chatbot_qa')\
                         .select('*')\
                         .eq('categoria', category_name)\
                         .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa/search")
async def search_qa(query: str):
    """Busca preguntas y respuestas que contengan la palabra clave en la pregunta o respuesta."""
    try:
        # Supabase no tiene una función de búsqueda de texto completo tan directa en .eq()
        # Para búsquedas más avanzadas, se recomienda usar funciones de PostgreSQL o trigram
        # Este es un ejemplo básico que filtra en el cliente.
        # Para producción, considera usar `ilike` o `fts` en Supabase.
        response = supabase.table('chatbot_qa').select('*').execute()
        if response.error:
            raise HTTPException(status_code=500, detail=response.error.message)
        
        results = [item for item in response.data if query.lower() in item['pregunta'].lower() or query.lower() in item['respuesta'].lower()]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
