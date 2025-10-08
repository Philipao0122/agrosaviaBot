import tkinter as tk
from tkinter import scrolledtext, messagebox
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import unicodedata
import re
import os

# Importar módulos personalizados
try:
    from text_normalizer import TextNormalizer
    from ubicacion_helper import UbicacionHelper, extraer_cultivo_desde_pregunta, validar_respuesta_ubicacion
except ImportError as e:
    print(f"Error al importar módulos personalizados: {e}")
    raise
# Cargar el modelo de spaCy una sola vez
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    messagebox.showerror("Error", "Modelo de spaCy 'es_core_news_md' no encontrado. Intentando descargarlo...")
    try:
        spacy.cli.download("es_core_news_md")
        nlp = spacy.load("es_core_news_md")
        messagebox.showinfo("Éxito", "Modelo de spaCy descargado e cargado exitosamente.")
    except Exception as e:
        nlp = None # Indicar que el modelo no está disponible

# Palabras que queremos conservar (normalizadas)
custom_stopwords_to_keep = {"no", "si", "donde", "cuando", "como", "que", "cual", "cuanto"}

def normalize_text(text):
    """
    Normaliza el texto para procesamiento, manteniendo caracteres especiales del español.
    
    Args:
        text (str): Texto a normalizar
        
    Returns:
        str: Texto normalizado en minúsculas, sin signos de puntuación ni espacios extra,
             manteniendo caracteres especiales del español (ñ, tildes, etc.)
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Si el modelo de spaCy no está disponible, hacer una normalización básica
    if not nlp:
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar signos de puntuación (excepto guiones y apóstrofes dentro de palabras)
        text = re.sub(r'(?<![\w\-\'])[^\w\s\-\']+|[^\w\s\-\']+(?![\w\-\'])', ' ', text)
        # Eliminar espacios extra
        return ' '.join(text.split())

    # Convertir a minúsculas
    text = text.lower()
    
    # Normalizar caracteres Unicode (combinar caracteres acentuados)
    text = unicodedata.normalize("NFC", text)
    
    # Procesar con spaCy
    doc = nlp(text)
    
    # Filtrar y lematizar tokens
    tokens = [
        token.lemma_
        for token in doc
        if not (token.is_stop and token.lemma_.lower() not in custom_stopwords_to_keep)
        and not token.is_punct
        and not token.is_space
    ]
    
    # Unir tokens en un solo string
    return ' '.join(tokens)

class ChatbotApp:
    def __init__(self, master):
        master.title("Chatbot MVP")
        master.geometry("700x500")

        # Inicializar atributos básicos
        self.df = None
        self.df_ubicacion = None
        self.df_numerico = None
        
        # Inicializar primero el normalizador de texto
        self.normalizer = TextNormalizer()
        
        # Inicializar el helper de ubicación después del normalizador
        self.ubicacion_helper = UbicacionHelper()
        
        try:
            # Cargar datos
            self.df = self.load_data("chatbot_datav5.csv")
            if self.df is None:
                self.cargar_datos()
                
        except Exception as e:
            messagebox.showerror("Error de Inicialización", 
                              f"Error al inicializar el chatbot: {str(e)}")
            raise
        
        # Listas para manejo de saludos y despedidas
        self.saludos = ["hola", "buenos días", "buenas tardes", "buenas noches", "saludos", "qué tal", "cómo estás"]
        self.respuestas_saludo = ["¡Hola! ¿En qué puedo ayudarte hoy?",
                               "¡Buen día! ¿Cómo estás?",
                               "¡Hola! ¿En qué puedo asistirte?",
                               "¡Saludos! ¿Cómo puedo ayudarte con información agrícola hoy?"]
        self.despedidas = ["adiós", "hasta luego", "hasta pronto", "nos vemos", "chao", "hasta la próxima"]
        self.respuestas_despedida = ["¡Hasta luego! Que tengas un excelente día.",
                                  "¡Nos vemos pronto! Si tienes más preguntas, aquí estaré.",
                                  "¡Hasta la próxima! Fue un placer ayudarte.",
                                  "¡Chao! Recuerda que estoy aquí para ayudarte con información agrícola."]
        
        # Configuración de la interfaz de usuario
        self.chat_history = scrolledtext.ScrolledText(master, wrap=tk.WORD, state='disabled', font=("Arial", 10))
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.user_input_frame = tk.Frame(master)
        self.user_input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.user_entry = tk.Entry(self.user_input_frame, font=("Arial", 10))
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_entry.bind("<Return>", self.send_message_event)

        self.send_button = tk.Button(self.user_input_frame, text="Enviar", command=self.send_message, font=("Arial", 10))
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # Botón para mostrar respuestas alternativas
        self.show_alternatives_button = tk.Button(
            self.user_input_frame, 
            text="Mostrar más opciones", 
            command=self.show_alternative_responses,
            font=("Arial", 10),
            state='disabled'  # Deshabilitado por defecto
        )
        self.show_alternatives_button.pack(side=tk.RIGHT, padx=5)
        self.alternative_responses = []  # Para almacenar respuestas alternativas

        self.display_message("Chatbot: ¡Hola! Soy tu asistente de análisis vegetal. ¿En qué puedo ayudarte hoy?", "chatbot")

    def load_data(self, filename):
        print("\n" + "="*80)
        print("INICIALIZACIÓN DEL CHATBOT - CARGA DE DATOS")
        print("="*80)
        
        try:
            # Cargar el archivo CSV base (general)
            print(f"\n[1/4] Cargando dataset principal: {filename}")
            df = pd.read_csv(filename)
            print(f"  ✓ Dataset principal cargado con éxito")
            print(f"  - Número de registros: {len(df)}")
            
            # Validar columnas requeridas en el dataset principal
            required_columns = ['pregunta', 'respuesta_compuesta', 'categoria']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                error_msg = f"El dataset principal no tiene las columnas requeridas: {', '.join(missing_columns)}"
                print(f"  ❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
                return None
            
            if len(df) > 0:
                print(f"  - Columnas disponibles: {', '.join(df.columns)}")
                print(f"  - Categorías únicas: {df['categoria'].nunique()}")
                print(f"  - Ejemplo de pregunta: '{df['pregunta'].iloc[0][:100]}...'")
            
            # Cargar los otros conjuntos de datos
            print("\n[2/4] Cargando datasets adicionales...")
            
            print("\n  - Dataset de ubicación (departamentos.csv):")
            self.df_ubicacion = self._load_additional_data("UbicacionyGeneral.csv")
            
            print("\n  - Dataset numérico (chatbot_data_optimizado.csv):")
            self.df_numerico = self._load_additional_data("chatbot_data_optimizado.csv")
            
            print("\n  - Dataset complejo (datos_complejos.csv):")
            self.df_complejo = self._load_additional_data("datos_complejos.csv")
            
            print("\n" + "="*50)
            print("RESUMEN DE DATOS CARGADOS")
            print("="*50)
            print(f"  • Dataset principal: {len(df)} registros")
            print(f"  • Dataset ubicación: {len(self.df_ubicacion) if self.df_ubicacion is not None else 'No disponible'}")
            print(f"  • Dataset numérico: {len(self.df_numerico) if self.df_numerico is not None else 'No disponible'}")
            print(f"  • Dataset complejo: {len(self.df_complejo) if self.df_complejo is not None else 'No disponible'}")
            print("="*50 + "\n")
            
            return df
            
        except FileNotFoundError as e:
            error_msg = f"Error: El archivo {filename} no se encontró. Asegúrate de que esté en el mismo directorio que el script."
            print(f"\n❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
            return None
            
        except Exception as e:
            error_msg = f"Error inesperado al cargar los datos: {str(e)}"
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            return None
            
    def _load_additional_data(self, filename):
        """Carga un archivo de datos adicional, retorna None si no existe."""
        try:
            if not os.path.exists(filename):
                print(f"  ✗ Archivo no encontrado: {filename}")
                return None
                
            print(f"  ✓ Cargando {filename}...")
            df = pd.read_csv(filename)
            
            # Verificar columnas requeridas
            required_columns = ['pregunta', 'respuesta_compuesta']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ⚠️ Advertencia: El archivo {filename} no tiene las columnas requeridas: {', '.join(missing_columns)}")
            else:
                print(f"  ✓ Dataset cargado correctamente con {len(df)} registros")
                if 'categoria' in df.columns:
                    print(f"  - Categorías únicas: {df['categoria'].nunique()}")
                print(f"  - Ejemplo de pregunta: '{df['pregunta'].iloc[0][:80]}...'")
                
            return df
            
        except Exception as e:
            error_msg = f"Error al cargar {filename}: {str(e)}"
            print(f"  ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_model(self):
        # Aplicar limpieza y normalización a las preguntas del dataset
        self.df["pregunta_limpia"] = self.df["pregunta"].apply(normalize_text)
        self.df["respuesta_compuesta_limpia"] = self.df["respuesta_compuesta"].apply(normalize_text)

        # Crear y entrenar el modelo TF-IDF y Naive Bayes
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            MultinomialNB()
        )
        self.model.fit(self.df["pregunta_limpia"], self.df["categoria"])

        # Saludos y despedidas
        self.saludos = ["hola", "buenos dias", "buenas tardes", "buenas noches", "hola que tal", "como estas", "ey"]
        self.respuestas_saludo = ["¡Hola! ¿En qué puedo ayudarte?", "¡Buenos días! ¿Qué necesitas?", "Hola, dime tu duda."]
        self.despedidas = ["gracias", "hasta luego", "adios", "nos vemos", "bye", "chao"]
        self.respuestas_despedida = ["¡Hasta pronto!", "Gracias por tu consulta. ¡Éxitos!", "Nos vemos."]

    def send_message_event(self, event):
        self.send_message()

    def send_message(self):
        user_text = self.user_entry.get()
        if not user_text.strip():
            return

        self.display_message(f"Tú: {user_text}", "user")
        self.user_entry.delete(0, tk.END)

        response = self.get_chatbot_response(user_text)
        self.display_message(f"Chatbot: {response}", "chatbot")

    def normalize_query(self, query):
        """Normaliza una consulta para mejorar la coincidencia."""
        if not query or not isinstance(query, str):
            return ""
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            # Si por alguna razón no hay normalizador, usamos una normalización básica
            return query.lower().strip()
        return self.normalizer.normalize_text(query)
        
    def get_question_type(self, query):
        """Determina el tipo de pregunta para seleccionar el conjunto de datos adecuado."""
        print("\n" + "="*80)
        print(f"ANÁLISIS DE PREGUNTA: '{query}'")
        print("="*80)
        
        # Reiniciar el cultivo actual
        self.cultivo_actual = None
        
        # Normalizar la consulta para el procesamiento
        query_normalized = self.normalize_query(query)
        print(f"Consulta normalizada: '{query_normalized}'")
        
        # Verificar si es una pregunta de ubicación
        query_lower = query_normalized.lower()
        
        # Palabras clave para identificar preguntas de ubicación
        ubicacion_keywords = [
            'dónde', 'donde', 'ubicación', 'ubicacion', 'lugar', 'sitio',
            'en qué lugar', 'en que lugar', 'en qué zona', 'en que zona',
            'en qué región', 'en que region', 'en qué regiones', 'en que regiones',
            'en qué departamento', 'en que departamento', 'en qué departamentos', 
            'en que departamentos', 'de dónde', 'de donde', 'procedencia', 
            'origen', 'localización', 'localizacion'
        ]
        
        # Verificar si es una pregunta de ubicación
        es_ubicacion = any(palabra in query_lower for palabra in ubicacion_keywords)
        
        if es_ubicacion:
            # Extraer el cultivo de la pregunta
            self.cultivo_actual = extraer_cultivo_desde_pregunta(query)
            if self.cultivo_actual:
                print(f"  - Cultivo detectado en pregunta de ubicación: {self.cultivo_actual}")
                return f'ubicacion_{self.cultivo_actual}'
            else:
                print("  - No se pudo identificar un cultivo específico en la pregunta de ubicación")
                return 'ubicacion'
        
        # Palabras clave para identificar el tipo de pregunta
        ubicacion_keywords = ['dónde', 'donde', 'ubicación', 'ubicacion', 'lugar', 'sitio',
                             'en qué lugar', 'en que lugar', 'en qué zona', 'en que zona',
                             'en qué región', 'en que region', 'en qué regiones', 'en que regiones',
                             'en qué departamento', 'en que departamento', 'en qué departamentos', 'en que departamentos',
                             'de dónde', 'de donde', 'procedencia', 'origen', 'localización', 'localizacion']
        
        # Nombres de departamentos y regiones de Colombia
        departamentos = [
            'amazonas', 'antioquia', 'arauca', 'atlántico', 'atlantico', 'bolívar', 'bolivar',
            'boyacá', 'boyaca', 'caldas', 'caquetá', 'caqueta', 'casanare', 'cauca', 'cesar',
            'chocó', 'choco', 'córdoba', 'cordoba', 'cundinamarca', 'guainía', 'guainia',
            'guaviare', 'huila', 'la guajira', 'magdalena', 'meta', 'nariño', 'nariÃ±o', 'narinio',
            'norte de santander', 'putumayo', 'quindío', 'quindio', 'risaralda', 'san andrés', 'san andres',
            'santander', 'sucre', 'tolima', 'valle del cauca', 'vaupés', 'vaupes', 'vichada'
        ]
        
        # Agregar departamentos a las palabras clave de ubicación
        ubicacion_keywords.extend(departamentos)
        
        numerico_keywords = ['cuánto', 'cuanto', 'porcentaje', '%', 'cantidad', 'número',
                           'numero', 'nÃºmero', 'nãºmero', 'medida', 'medidas', 'medición',
                           'medicion', 'mediciÃ³n', 'mediciã³n', 'tamaño', 'tamano', 'tamaÃ±o',
                           'tamaã±o', 'peso', 'volumen', 'área', 'area', 'Ã¡rea', 'ã¡rea',
                           'longitud', 'ancho', 'alto', 'profundidad', 'densidad', 'ph', 'ph.',
                           'ph ', ' ph', 'nivel', 'niveles', 'rango', 'rango de', 'entre',
                           'promedio', 'media', 'máximo', 'maximo', 'mÃ¡ximo', 'mã¡ximo',
                           'mínimo', 'minimo', 'mÃ­nimo', 'mã­nimo', 'cuantos', 'cuántos',
                           'cuantas', 'cuántas', 'cuantos', 'cuántos']
        
        # Verificar palabras clave de ubicación
        ubicacion_matches = [kw for kw in ubicacion_keywords if kw in query_lower]
        if ubicacion_matches:
            print(f"Tipo: Ubicación (palabras clave encontradas: {', '.join(ubicacion_matches)})")
            
            # Si la pregunta es sobre ubicación, verificar si menciona un cultivo específico
            cultivos = ['plátano', 'platano', 'arroz', 'maíz', 'maiz', 'café', 'cafe', 'yuca', 
                       'papa', 'fríjol', 'frijol', 'aguacate', 'palma', 'caña', 'cana', 'banano',
                       'cacao', 'algodón', 'algodon', 'soya', 'soja', 'trigo', 'cebada', 'cebolla',
                       'tomate', 'pimiento', 'zanahoria', 'lechuga', 'repollo', 'brócoli', 'brocoli',
                       'espinaca', 'acelga', 'remolacha', 'rábano', 'rabanito', 'rábano', 'rabanito',
                       'apio', 'perejil', 'cilantro', 'albahaca', 'menta', 'hierbabuena', 'orégano', 'oregano']
            
            cultivo_encontrado = next((cultivo for cultivo in cultivos if cultivo in query_lower), None)
            if cultivo_encontrado:
                print(f"  - Cultivo detectado: {cultivo_encontrado}")
                return f'ubicacion_{cultivo_encontrado}'
                
            return 'ubicacion'
            
        # Verificar palabras clave numéricas
        numerico_matches = [kw for kw in numerico_keywords if kw in query_lower]
        if numerico_matches:
            print(f"Tipo: Numérico (palabras clave encontradas: {', '.join(numerico_matches)})")
            return 'numerico'
            
        # Verificar preguntas complejas
        word_count = len(query.split())
        has_question_mark = '?' in query
        if word_count > 10 or has_question_mark:
            reason = []
            if word_count > 10:
                reason.append(f"muchas palabras ({word_count} > 10)")
            if has_question_mark:
                reason.append("contiene signo de interrogación")
            print(f"Tipo: Complejo ({', '.join(reason)})")
            return 'complejo'
            
        print("Tipo: General (ninguna categoría específica detectada)")
        return 'general'
    
    def get_best_match(self, query, df, dataset_name="", return_match_info=False):
        """
        Encuentra la mejor coincidencia en el dataframe usando búsqueda semántica y por palabras clave.
        
        Args:
            query: La pregunta del usuario
            df: DataFrame con las preguntas y respuestas
            dataset_name: Nombre del dataset para logging
            return_match_info: Si es True, retorna un diccionario con información detallada
            
        Returns:
            Si return_match_info es False: La mejor respuesta encontrada o None
            Si return_match_info es True: Un diccionario con información detallada de la mejor coincidencia
        """
        print(f"\nBuscando en el dataset: {dataset_name or 'general'}")
        
        # Normalizar la consulta
        original_query = query
        query = self.normalize_query(query)
        print(f"Consulta normalizada: '{query}'")
        
        # Crear una copia normalizada del dataframe para búsqueda
        df_search = df.copy()
        
        # Normalizar las columnas de búsqueda si existen
        for col in ['pregunta', 'respuesta_compuesta', 'respuesta', 'categoria']:
            if col in df_search.columns:
                df_search[f'{col}_normalized'] = df_search[col].fillna('').apply(self.normalize_query)
        
        if df is None:
            print("  - El dataset no está disponible")
            return None
            
        if df.empty:
            print("  - El dataset está vacío")
            return None
            
        # Hacer una copia del dataframe para no modificar el original
        df = df.copy()
        
        # 1. Primero intentar búsqueda exacta o por palabras clave
        query_lower = query.lower()
        
        # Identificar palabras clave importantes (excluyendo palabras comunes)
        palabras_comunes = {'como', 'para', 'con', 'del', 'las', 'los', 'que', 'cual', 'cuales', 'donde', 'cuando', 'esta', 'estan', 'está', 'están'}
        palabras_clave = [p for p in query_lower.split() 
                         if len(p) > 2 and p not in palabras_comunes]
        
        # Si no hay palabras clave suficientemente específicas, usar todas las palabras
        if not palabras_clave:
            palabras_clave = [p for p in query_lower.split() if len(p) > 1]
        
        print(f"  - Palabras clave identificadas: {', '.join(palabras_clave) if palabras_clave else 'Ninguna'}")
        
        # Buscar coincidencias en las columnas relevantes (usando versiones normalizadas si existen)
        columnas_busqueda = []
        if 'pregunta_normalized' in df_search.columns:
            columnas_busqueda.append(('pregunta_normalized', 3))  # Peso 3 para pregunta
        if 'respuesta_compuesta_normalized' in df_search.columns:
            columnas_busqueda.append(('respuesta_compuesta_normalized', 2))  # Peso 2 para respuesta_compuesta
        
        # Añadir columnas adicionales con peso 1
        for col in ['respuesta_normalized', 'categoria_normalized', 'departamento_normalized', 'municipio_normalized']:
            if col in df_search.columns and col not in [c[0] for c in columnas_busqueda]:
                columnas_busqueda.append((col, 1))
        
        # Si no hay columnas normalizadas, usar las originales
        if not columnas_busqueda:
            columnas_busqueda = [
                ('pregunta', 3),
                ('respuesta_compuesta', 2),
                ('respuesta', 1),
                ('categoria', 1),
                ('departamento', 1),
                ('municipio', 1)
            ]
            columnas_busqueda = [(col, weight) for col, weight in columnas_busqueda if col in df_search.columns]
        
        print(f"  - Columnas de búsqueda: {', '.join([f'{col} (peso: {weight})' for col, weight in columnas_busqueda])}")
        
        # Inicializar puntuación de coincidencia
        df_search['puntaje_coincidencia'] = 0
        
        # Ponderar más las coincidencias en la pregunta que en la respuesta
        for palabra in palabras_clave:
            for col, weight in columnas_busqueda:
                if col in df_search.columns:
                    df_search['puntaje_coincidencia'] += df_search[col].str.contains(
                        re.escape(palabra), case=False, na=False, regex=True
                    ).astype(int) * weight
                    
                    # Búsqueda parcial para palabras compuestas
                    if len(palabra) > 4:  # Solo para palabras medianas/largas
                        df_search['puntaje_coincidencia'] += df_search[col].str.contains(
                            f"\\b{palabra[:4]}", case=False, na=False, regex=True
                        ).astype(int) * (weight / 2)  # Mitad de peso para coincidencias parciales
        
        # Filtrar solo las filas con al menos una coincidencia
        mascara = df_search['puntaje_coincidencia'] > 0
        
        # Si no hay coincidencias, retornar None
        if not mascara.any():
            print("  - No se encontraron coincidencias con las palabras clave")
            return None
        
        # Si encontramos coincidencias, procesarlas
        if mascara.any():
            df_coincidencias = df_search[mascara].copy()
            print(f"  - Se encontraron {len(df_coincidencias)} coincidencias por palabras clave")
            
            # Ordenar por puntaje de coincidencia (ya calculado)
            df_coincidencias = df_coincidencias.sort_values('puntaje_coincidencia', ascending=False)
            
            # Mostrar las 3 mejores coincidencias para depuración
            print("  - Mejores coincidencias encontradas:")
            for i, (_, row) in enumerate(df_coincidencias.head(3).iterrows(), 1):
                print(f"    {i}. Puntaje: {row['puntaje_coincidencia']:.1f} - "
                      f"Pregunta: {row.get('pregunta', '')[:60]}...")
            
            # Filtrar respuestas que contengan "NO INDICA" en cualquier campo relevante
            columnas_verificar = ['respuesta_compuesta', 'respuesta', 'pregunta', 'departamento', 'municipio']
            columnas_verificar = [c for c in columnas_verificar if c in df_coincidencias.columns]
            
            # Crear máscara para excluir filas con "NO INDICA" en cualquier campo relevante
            mascara_no_indica = pd.Series(False, index=df_coincidencias.index)
            for col in columnas_verificar:
                mascara_no_indica = mascara_no_indica | df_coincidencias[col].astype(str).str.contains('NO INDICA', case=False, na=False)
            
            # Filtrar las coincidencias
            sin_no_indica = df_coincidencias[~mascara_no_indica]
            
            # Si después de filtrar aún hay coincidencias, usarlas
            if not sin_no_indica.empty:
                print(f"  - Se encontraron {len(sin_no_indica)} coincidencias válidas después de filtrar 'NO INDICA'")
                df_coincidencias = sin_no_indica
            else:
                # Si no hay coincidencias sin "NO INDICA", usar las que tienen el puntaje más alto
                print("  - Todas las coincidencias contienen 'NO INDICA', usando la mejor disponible")
                df_coincidencias = df_coincidencias.head(1)
            
            # Verificar que la mejor coincidencia tenga un puntaje mínimo
            if not df_coincidencias.empty:
                mejor_row = df_coincidencias.iloc[0]
                umbral_minimo = max(1, len(palabras_clave) * 0.7)  # Al menos el 70% de las palabras clave
                
                if mejor_row['puntaje_coincidencia'] >= umbral_minimo:
                    print(f"  - Mejor coincidencia con puntaje {mejor_row['puntaje_coincidencia']:.1f} "
                          f"(umbral mínimo: {umbral_minimo:.1f})")
                else:
                    print(f"  - Mejor coincidencia con puntaje {mejor_row['puntaje_coincidencia']:.1f} "
                          f"por debajo del umbral mínimo de {umbral_minimo:.1f}")
                    return None
            else:
                print("  - No hay coincidencias válidas después de filtrar")
                return None
            
            if return_match_info:
                # Almacenar hasta 4 respuestas alternativas (omitiendo la primera que es la principal)
                self.alternative_responses = []
                if len(df_coincidencias) > 1:
                    self.alternative_responses = df_coincidencias.iloc[1:5].apply(
                        lambda row: row.get('respuesta_compuesta', row.get('respuesta', 'Sin respuesta')), 
                        axis=1
                    ).tolist()
                    # Habilitar el botón si hay alternativas
                    self.show_alternatives_button.config(state='normal' if self.alternative_responses else 'disabled')
                else:
                    self.show_alternatives_button.config(state='disabled')
                
                return {
                    'answer': mejor_row.get('respuesta_compuesta', mejor_row.get('respuesta', 'Sin respuesta')),
                    'similarity': 1.0,  # Máxima similitud para coincidencias exactas
                    'question': mejor_row.get('pregunta', ''),
                    'dataset': dataset_name
                }
            return mejor_row.get('respuesta_compuesta', mejor_row.get('respuesta', 'Sin respuesta'))
        
        # 2. Si no hay coincidencias exactas, usar búsqueda semántica
        print("  - No se encontraron coincidencias exactas, intentando búsqueda semántica...")
        
        # Normalizar la pregunta del usuario
        pregunta_limpia = normalize_text(query)
        if not pregunta_limpia:
            print("  - No se pudo normalizar la pregunta")
            return None
            
        print(f"  - Pregunta normalizada: '{pregunta_limpia}'")
        
        try:
            # Vectorizar la pregunta del usuario
            vectorizer = self.model.named_steps["tfidfvectorizer"]
            
            # Asegurarse de que todas las preguntas estén normalizadas
            df["pregunta_limpia"] = df["pregunta"].apply(
                lambda x: normalize_text(str(x)) if pd.notna(x) else ""
            )
            
            # Filtrar preguntas vacías
            df = df[df["pregunta_limpia"] != ""]
            
            if df.empty:
                print("  - No hay preguntas válidas para comparar")
                return None
                
            # Vectorizar todas las preguntas del dataset de una vez
            preguntas_vec = vectorizer.transform(df["pregunta_limpia"])
            pregunta_vec = vectorizer.transform([pregunta_limpia])
            
            # Calcular similitud con todas las preguntas
            similitudes = cosine_similarity(pregunta_vec, preguntas_vec)[0]
            
            # Encontrar la mejor coincidencia, ignorando las que contengan "NO INDICA"
            mejor_idx = -1
            mejor_similitud = 0
            mejor_row = None
            
            # Buscar la mejor coincidencia que no contenga "NO INDICA"
            for i in range(len(similitudes)):
                if similitudes[i] > mejor_similitud:
                    row = df.iloc[i]
                    respuesta = str(row.get('respuesta_compuesta', row.get('respuesta', '')))
                    if 'NO INDICA' not in respuesta.upper():
                        mejor_similitud = similitudes[i]
                        mejor_idx = i
                        mejor_row = row
            
            # Si no se encontró ninguna coincidencia sin "NO INDICA", usar la mejor disponible
            if mejor_idx == -1 and len(similitudes) > 0:
                print("  - Todas las coincidencias contienen 'NO INDICA', mostrando la mejor coincidencia")
                mejor_idx = similitudes.argmax()
                mejor_similitud = similitudes[mejor_idx]
                mejor_row = df.iloc[mejor_idx]
            
            if mejor_row is not None:
                print(f"  - Mejor coincidencia: {mejor_similitud*100:.1f}% de similitud")
                
                if return_match_info:
                    return {
                        'answer': mejor_row.get('respuesta_compuesta', 'Respuesta no disponible'),
                        'similarity': float(mejor_similitud),
                        'question': mejor_row.get('pregunta', ''),
                        'dataset': dataset_name
                    }
                elif mejor_similitud >= 0.4:  # 40% de similitud
                    return mejor_row.get('respuesta_compuesta', 'Respuesta no disponible')
                else:
                    print(f"  ✗ La mejor coincidencia no supera el umbral del 40% ({mejor_similitud*100:.1f}%)")
                    return None
            else:
                print("  - No se encontraron coincidencias válidas")
                return None
                
        except Exception as e:
            print(f"  ✗ Error al buscar coincidencia: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_chatbot_response(self, query):
        print("\n" + "="*120)
        print(f"🚀 INICIO DE PROCESAMIENTO DE PREGUNTA")
        print(f"   Pregunta: '{query}'")
        print("="*120)
        
        # Verificar si es una pregunta de ubicación (contiene 'donde' o 'dónde')
        query_lower = query.lower()
        es_consulta_ubicacion = any(palabra in query_lower for palabra in ['donde', 'dónde'])
        
        if es_consulta_ubicacion and self.df_ubicacion is not None:
            print("🔍 DETECTADA PREGUNTA DE UBICACIÓN")
            
            # Primero, buscar en el dataset general de ubicaciones (que tiene más registros)
            print("  - Buscando en el dataset general de ubicaciones...")
            general_match = self.get_best_match(query, self.df_ubicacion, "ubicación_general", return_match_info=True)
            
            # Luego, verificar si hay un cultivo específico mencionado
            cultivo = self.ubicacion_helper.extraer_cultivo(query)
            specific_match = None
            
            if cultivo:
                print(f"  - Cultivo detectado: {cultivo}")
                # Buscar coincidencias específicas para este cultivo
                df_filtrado = self.df_ubicacion[
                    (self.df_ubicacion['pregunta'].str.lower().str.contains(cultivo.lower(), na=False) |
                     self.df_ubicacion['respuesta_compuesta'].str.lower().str.contains(cultivo.lower(), na=False))
                ]
                
                if not df_filtrado.empty:
                    print(f"  - Se encontraron {len(df_filtrado)} registros específicos para {cultivo}")
                    specific_match = self.get_best_match(query, df_filtrado, "ubicación_específica", return_match_info=True)
            
            # Decidir qué resultado usar
            if specific_match and specific_match.get('similarity', 0) >= 0.5:  # Umbral más alto para coincidencias específicas
                print("  - Usando resultado específico del cultivo")
                return specific_match['answer']
            elif general_match:
                print("  - Usando resultado del dataset general de ubicaciones")
                return general_match['answer']
                
            print("  - No se encontraron coincidencias en los datasets de ubicación")
        
        query_lower = query.lower()
        
        # Manejar saludos
        saludos_encontrados = [s for s in self.saludos if s in query_lower]
        if saludos_encontrados:
            print(f"Saludo detectado: {saludos_encontrados}")
            return random.choice(self.respuestas_saludo)
        
        # Manejar despedidas
        despedidas_encontradas = [d for d in self.despedidas if d in query_lower]
        if despedidas_encontradas:
            print(f"Despedida detectada: {despedidas_encontradas}")
            return random.choice(self.respuestas_despedida)
        
        # Lista de conjuntos de datos a buscar, en orden de prioridad
        datasets = [
            (self.df, "principal"),
            (self.df_ubicacion, "ubicación"),
            (self.df_numerico, "numérico"),
            (self.df_complejo, "complejo")
        ]
        
        # Primero, intentar con el dataset específico según el tipo de pregunta
        question_type = self.get_question_type(query)
        
        # Si es una pregunta de ubicación con cultivo específico, buscar primero en el dataset de ubicación
        if question_type.startswith('ubicacion_') and self.df_ubicacion is not None:
            cultivo = question_type.split('_', 1)[1]
            print(f"\n🔍 BUSCANDO UBICACIÓN PARA CULTIVO: {cultivo.upper()}")
            
            # Primero intentar con el dataset de ubicación
            if self.df_ubicacion is not None:
                # Filtrar por el cultivo en las preguntas o respuestas
                mask = (self.df_ubicacion['pregunta'].str.lower().str.contains(cultivo, na=False)) | \
                       (self.df_ubicacion['respuesta_compuesta'].str.lower().str.contains(cultivo, na=False))
                df_filtrado = self.df_ubicacion[mask].copy()
                
                if not df_filtrado.empty:
                    print(f"  - Se encontraron {len(df_filtrado)} registros relacionados con {cultivo}")
                    # Usar el dataset filtrado para la búsqueda
                    match_info = self.get_best_match(query, df_filtrado, f"ubicación_{cultivo}", return_match_info=True)
                    if match_info and match_info['similarity'] >= 0.4:  # 40% de similitud
                        return match_info['answer']
                    
                    # Si no se encontró una buena coincidencia, continuar con la búsqueda normal
                    print(f"  - No se encontró una respuesta específica para {cultivo}, continuando con búsqueda general")
        
        # Configurar el orden de búsqueda según el tipo de pregunta
        if question_type == 'ubicacion' and self.df_ubicacion is not None:
            datasets.insert(0, (self.df_ubicacion, "ubicación"))
        elif question_type == 'numerico' and self.df_numerico is not None:
            datasets.insert(0, (self.df_numerico, "numérico"))
        elif question_type == 'complejo' and self.df_complejo is not None:
            datasets.insert(1, (self.df_complejo, "complejo"))
        
        # Eliminar duplicados manteniendo el orden
        seen = set()
        unique_datasets = []
        for df, name in datasets:
            if df is not None and not any(df.equals(existing_df) for existing_df, _ in unique_datasets):
                unique_datasets.append((df, name))
        datasets = unique_datasets
        
        # Mostrar información de búsqueda
        print("\n" + "-"*80)
        print("🔍 INICIANDO BÚSQUEDA EN DATASETS")
        print(f"  - Datasets disponibles: {', '.join([name for df, name in datasets])}")
        print("-"*80)
        
        # Buscar en todos los datasets y guardar los resultados
        all_matches = []
        
        for df, nombre in datasets:
            print(f"\nBuscando en dataset {nombre}...")
            if df is not None and not df.empty:
                print(f"  - Total de registros: {len(df)}")
                if 'categoria' in df.columns:
                    print(f"  - Categorías disponibles: {df['categoria'].nunique()}")
                
                # Obtener la mejor coincidencia de este dataset
                best_match = self.get_best_match(query, df, nombre, return_match_info=True)
                if best_match and best_match['similarity'] > 0:  # Solo considerar coincidencias con similitud > 0
                    all_matches.append(best_match)
                    print(f"  - Mejor coincidencia en {nombre}: {best_match['similarity']*100:.1f}%")
                else:
                    print(f"  - No se encontraron coincidencias útiles en el dataset {nombre}")
            else:
                print(f"  - Dataset {nombre} está vacío o no disponible")
        
        # Seleccionar la mejor coincidencia de todos los datasets
        if all_matches:
            # Ordenar por similitud descendente
            all_matches.sort(key=lambda x: x['similarity'], reverse=True)
            best_overall = all_matches[0]
            
            print("\n" + "="*80)
            print(f"🏆 MEJOR COINCIDENCIA GLOBAL ({best_overall['dataset']} - {best_overall['similarity']*100:.1f}%)")
            print("="*80)
            print(f"Pregunta del usuario: {query}")
            print(f"Pregunta coincidente: {best_overall['question']}")
            print(f"Respuesta: {best_overall['answer']}")
            print("="*80 + "\n")
            return best_overall['answer']
        
        # Si llegamos aquí, no se encontró ninguna respuesta adecuada
        print("\n⚠️ No se encontró una respuesta adecuada en ningún dataset")
        return "Lo siento, no tengo suficiente información para responder a tu pregunta. ¿Podrías reformularla o proporcionar más detalles?"

    def display_message(self, message, sender):
        self.chat_history.config(state=tk.NORMAL)
        if sender == "user":
            self.chat_history.insert(tk.END, f"{message}\n", "user_tag")
        else:
            self.chat_history.insert(tk.END, f"{message}\n", "chatbot_tag")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)
        
    def show_alternative_responses(self):
        """Muestra las respuestas alternativas en el chat."""
        if not self.alternative_responses:
            return
            
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, "\nOtras opciones de respuesta:\n", "chatbot_tag")
        
        for i, resp in enumerate(self.alternative_responses, 1):
            self.chat_history.insert(tk.END, f"{i}. {resp}\n", "chatbot_tag")
        
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.alternative_responses = []  # Limpiar después de mostrar
        self.show_alternatives_button.config(state='disabled')

        # Configurar tags para estilos
        self.chat_history.tag_config("user_tag", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_history.tag_config("chatbot_tag", foreground="green", font=("Arial", 10))


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()

