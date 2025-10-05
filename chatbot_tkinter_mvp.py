
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
        self.master = master
        master.title("Chatbot MVP")
        master.geometry("700x500")

        self.df = self.load_data("chatbot_datav5.csv")
        if self.df is None:
            master.destroy()
            return
        
        if nlp is None:
            messagebox.showerror("Error", "El chatbot no puede funcionar sin el modelo de spaCy. Cerrando aplicación.")
            master.destroy()
            return

        self.prepare_model()

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
            self.df_ubicacion = self._load_additional_data("departamentos_600.csv")
            
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

    def get_question_type(self, query):
        """Determina el tipo de pregunta para seleccionar el conjunto de datos adecuado."""
        print("\n" + "="*80)
        print(f"ANÁLISIS DE PREGUNTA: '{query}'")
        print("="*80)
        query_lower = query.lower()
        
        # Palabras clave para identificar el tipo de pregunta
        ubicacion_keywords = ['dónde', 'donde', 'ubicación', 'ubicacion', 'lugar', 'sitio',
                             'antioquia', 'valle del cauca', 'risaralda', 'cundinamarca',
                             'santander', 'nariño', 'nariÃ±o', 'narinio', 'nario',
                             'norte de santander', 'tolima', 'huila', 'cauca', 'valle',
                             'quindío', 'quindio', 'quindÃ­o', 'quindo', 'boyacá', 'boyaca',
                             'boyacÃ¡', 'casanare', 'meta', 'arauca', 'vichada', 'guainía',
                             'guainia', 'guainÃ­a', 'guaviare', 'vaupés', 'vaupes', 'vaupÃ©s',
                             'amazonas', 'guajira', 'la guajira', 'magdalena', 'cesar', 'césar',
                             'cesÃ¡r', 'cesã¡r', 'cÃ©sar', 'cã©sar', 'bolívar', 'bolivar',
                             'bolÃ­var', 'bolã­var', 'sucre', 'córdoba', 'cordoba', 'cã³rdoba',
                             'cã³rdoba', 'chocó', 'choco', 'chocÃ³', 'chocã³']
        
        numerico_keywords = ['cuánto', 'cuanto', 'porcentaje', '%', 'cantidad', 'número',
                           'numero', 'nÃºmero', 'nãºmero', 'medida', 'medidas', 'medición',
                           'medicion', 'mediciÃ³n', 'mediciã³n', 'tamaño', 'tamano', 'tamaÃ±o',
                           'tamaã±o', 'peso', 'volumen', 'área', 'area', 'Ã¡rea', 'ã¡rea',
                           'longitud', 'ancho', 'alto', 'profundidad', 'densidad', 'ph', 'ph.',
                           'ph ', ' ph', 'nivel', 'niveles', 'rango', 'rango de', 'entre',
                           'promedio', 'media', 'máximo', 'maximo', 'mÃ¡ximo', 'mã¡ximo',
                           'mínimo', 'minimo', 'mÃ­nimo', 'mã­nimo']
        
        # Verificar palabras clave de ubicación
        ubicacion_matches = [kw for kw in ubicacion_keywords if kw in query_lower]
        if ubicacion_matches:
            print(f"Tipo: Ubicación (palabras clave encontradas: {', '.join(ubicacion_matches)})")
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
        palabras_clave = [p for p in query_lower.split() if len(p) > 3]  # Filtrar palabras muy cortas
        
        # Buscar coincidencias exactas o parciales en las columnas relevantes
        columnas_busqueda = ['pregunta', 'respuesta', 'respuesta_compuesta', 'categoria']
        columnas_busqueda = [c for c in columnas_busqueda if c in df.columns]
        
        # Crear máscara de coincidencias
        mascara = pd.Series(False, index=df.index)
        for col in columnas_busqueda:
            for palabra in palabras_clave:
                # Buscar la palabra en cada celda (ignorando mayúsculas/minúsculas)
                mascara = mascara | df[col].str.contains(palabra, case=False, na=False, regex=False)
        
        # Si encontramos coincidencias exactas, procesarlas
        if mascara.any():
            df_coincidencias = df[mascara].copy()
            print(f"  - Se encontraron {len(df_coincidencias)} coincidencias por palabras clave")
            
            # Si hay múltiples coincidencias, ordenar por relevancia
            if len(df_coincidencias) > 0:
                # Contar el número de palabras clave que coinciden
                def contar_coincidencias(serie):
                    return serie.apply(lambda x: sum(1 for p in palabras_clave if p in str(x).lower()))
                
                # Usar map en lugar de applymap (que está obsoleto)
                df_coincidencias['puntaje'] = df_coincidencias[columnas_busqueda].apply(contar_coincidencias).sum(axis=1)
                df_coincidencias = df_coincidencias.sort_values('puntaje', ascending=False)
                
                # Filtrar respuestas que contengan "NO INDICA" en cualquier campo relevante
                columnas_verificar = ['respuesta_compuesta', 'respuesta', 'pregunta', 'departamento', 'municipio']
                columnas_verificar = [c for c in columnas_verificar if c in df_coincidencias.columns]
                
                # Crear máscara para excluir filas con "NO INDICA" en cualquier campo relevante
                mascara_no_indica = pd.Series(False, index=df_coincidencias.index)
                for col in columnas_verificar:
                    mascara_no_indica = mascara_no_indica | df_coincidencias[col].str.contains('NO INDICA', case=False, na=False)
                
                # Filtrar las coincidencias
                sin_no_indica = df_coincidencias[~mascara_no_indica]
                
                # Si después de filtrar aún hay coincidencias, usarlas
                if not sin_no_indica.empty:
                    print(f"  - Se encontraron {len(sin_no_indica)} coincidencias válidas después de filtrar 'NO INDICA'")
                    df_coincidencias = sin_no_indica
                else:
                    print("  - Todas las coincidencias contienen 'NO INDICA' en algún campo relevante")
                    return None
                
            if len(df_coincidencias) > 0:
                mejor_idx = df_coincidencias.index[0]
                mejor_row = df_coincidencias.iloc[0]
            else:
                print("  - No hay coincidencias válidas después de filtrar")
                return None
            
            if return_match_info:
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
        if question_type == 'ubicacion' and self.df_ubicacion is not None:
            datasets.insert(1, (self.df_ubicacion, "ubicación"))
        elif question_type == 'numerico' and self.df_numerico is not None:
            datasets.insert(1, (self.df_numerico, "numérico"))
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
            print(f"Pregunta: {best_overall['question']}")
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

        # Configurar tags para estilos
        self.chat_history.tag_config("user_tag", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_history.tag_config("chatbot_tag", foreground="green", font=("Arial", 10))


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()

