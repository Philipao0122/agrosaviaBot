
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
        messagebox.showerror("Error", f"No se pudo descargar ni cargar el modelo de spaCy: {e}. Asegúrate de tener conexión a internet y de que el modelo esté disponible.")
        nlp = None # Indicar que el modelo no está disponible

# Palabras que queremos conservar (normalizadas)
custom_stopwords_to_keep = {"no", "si", "donde", "cuando", "como", "que", "cual", "cuanto"}

def normalize_text(text):
    if not nlp: # Si el modelo de spaCy no se cargó, retornar el texto sin normalizar
        return text.lower()

    # Paso 1: Convertir a minúsculas
    text = text.lower()

    # Paso 2: Eliminar tildes y ñ
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")  # elimina tildes y ñ

    # Paso 3: Eliminar signos de puntuación
    text = re.sub(r"[^\w\s]", "", text)

    # Paso 4: Eliminar espacios extra
    text = re.sub(r"\s+", " ", text).strip()

    # Paso 5: Procesar con spaCy
    doc = nlp(text)

    # Paso 6: Lematizar, eliminar stopwords y quitar tildes también a los lemas
    tokens = [
        ''.join(
            c for c in unicodedata.normalize("NFD", token.lemma_)
            if unicodedata.category(c) != 'Mn'
        )
        for token in doc
        if not (token.is_stop and token.lemma_.lower() not in custom_stopwords_to_keep)
    ]

    return " ".join(tokens)

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
            if len(df) > 0:
                print(f"  - Columnas disponibles: {', '.join(df.columns)}")
                print(f"  - Categorías únicas: {df['categoria'].nunique()}")
                print(f"  - Ejemplo de pregunta: '{df['pregunta'].iloc[0][:100]}...'")
            
            # Cargar los otros conjuntos de datos
            print("\n[2/4] Cargando datasets adicionales...")
            
            print("\n  - Dataset de ubicación (datos_ubicacion.csv):")
            self.df_ubicacion = self._load_additional_data("datos_ubicacion.csv")
            
            print("\n  - Dataset numérico (chatbot_data_numerico.csv):")
            self.df_numerico = self._load_additional_data("chatbot_data_numerico.csv")
            
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
    
    def get_best_match(self, query, df, dataset_name=""):
        """Encuentra la mejor coincidencia en el dataframe dado."""
        print(f"\nBuscando en el dataset: {dataset_name or 'general'}")
        
        if df is None:
            print("  - El dataset no está disponible")
            return None
            
        if df.empty:
            print("  - El dataset está vacío")
            return None
            
        # Normalizar la pregunta del usuario
        pregunta_limpia = normalize_text(query)
        if not pregunta_limpia:
            print("  - No se pudo normalizar la pregunta")
            return None
            
        print(f"  - Pregunta normalizada: '{pregunta_limpia}'")
        
        # Vectorizar la pregunta del usuario y las preguntas del dataset
        vectorizer = self.model.named_steps["tfidfvectorizer"]
        try:
            pregunta_vec = vectorizer.transform([pregunta_limpia])
            
            # Calcular similitud con todas las preguntas en el dataset
            similitudes = []
            for idx, row in df.iterrows():
                pregunta_dataset = normalize_text(row['pregunta'] if 'pregunta' in row else '')
                if not pregunta_dataset:
                    continue
                    
                # Vectorizar la pregunta del dataset
                pregunta_dataset_vec = vectorizer.transform([pregunta_dataset])
                
                # Calcular similitud del coseno
                similitud = cosine_similarity(pregunta_vec, pregunta_dataset_vec)[0][0]
                similitudes.append((similitud, idx, row))
            
            if not similitudes:
                print("  - No se encontraron preguntas para comparar")
                return None
                
            # Ordenar por similitud descendente
            similitudes.sort(reverse=True, key=lambda x: x[0])
            
            # Imprimir las 3 mejores coincidencias
            print("  - Mejores coincidencias:")
            for i, (sim, idx, row) in enumerate(similitudes[:3]):
                print(f"    {i+1}. Similitud: {sim*100:.1f}% - '{row.get('pregunta', '')[:100]}{'...' if len(str(row.get('pregunta', ''))) > 100 else ''}")
            
            # Retornar la mejor coincidencia si supera el umbral
            mejor_sim, mejor_idx, mejor_row = similitudes[0]
            if mejor_sim >= 0.6:  # 60% de similitud
                print(f"  ✓ Mejor coincidencia seleccionada con {mejor_sim*100:.1f}% de similitud")
                return mejor_row.get('respuesta_compuesta', 'Respuesta no disponible')
            else:
                print(f"  ✗ Ninguna coincidencia superó el umbral del 60% (mejor: {mejor_sim*100:.1f}%)")
                
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
            
        # Determinar el tipo de pregunta
        question_type = self.get_question_type(query)
        
        # Seleccionar el conjunto de datos adecuado
        if question_type == 'ubicacion':
            print("\nBuscando en dataset de ubicación...")
            if self.df_ubicacion is not None:
                respuesta = self.get_best_match(query, self.df_ubicacion, "ubicación")
                if respuesta:
                    return respuesta
                print("  - No se encontró una respuesta adecuada en el dataset de ubicación")
            else:
                print("  - Dataset de ubicación no disponible")
                
        elif question_type == 'numerico':
            print("\nBuscando en dataset numérico...")
            if self.df_numerico is not None:
                respuesta = self.get_best_match(query, self.df_numerico, "numérico")
                if respuesta:
                    return respuesta
                print("  - No se encontró una respuesta adecuada en el dataset numérico")
            else:
                print("  - Dataset numérico no disponible")
                
        elif question_type == 'complejo':
            print("\nBuscando en dataset de preguntas complejas...")
            if self.df_complejo is not None:
                respuesta = self.get_best_match(query, self.df_complejo, "complejo")
                if respuesta:
                    return respuesta
                print("  - No se encontró una respuesta adecuada en el dataset de preguntas complejas")
            else:
                print("  - Dataset de preguntas complejas no disponible")
                
        # Si no se encontró una respuesta en los conjuntos específicos o no superó el umbral,
        # intentar con el conjunto general
        print("\n" + "-"*80)
        print("🔍 BUSCANDO EN EL DATASET GENERAL (chatbot_datav5.csv)")
        print("-"*80)
        try:
            print(f"  - Total de respuestas disponibles: {len(self.df)}")
            print(f"  - Columnas disponibles: {', '.join(self.df.columns)}")
            if 'categoria' in self.df.columns:
                print(f"  - Categorías disponibles: {', '.join(self.df['categoria'].unique().tolist())}")
                
            # Mostrar estadísticas de las preguntas
            if not self.df.empty:
                longitudes = self.df['pregunta'].str.len()
                print(f"  - Longitud promedio de preguntas: {longitudes.mean():.1f} caracteres")
                print(f"  - Pregunta más corta: '{self.df.loc[longitudes.idxmin(), 'pregunta']}'")
                print(f"  - Pregunta más larga: '{self.df.loc[longitudes.idxmax(), 'pregunta']}'")
                
                # Mostrar ejemplos de preguntas
                print("\n  Ejemplos de preguntas en el dataset:")
                for i, q in enumerate(self.df.sample(min(3, len(self.df)))['pregunta'].tolist(), 1):
                    print(f"    {i}. {q[:100]}{'...' if len(q) > 100 else ''}")
                print()
            # Normalizar la pregunta del usuario
            pregunta_limpia = normalize_text(query)
            if not pregunta_limpia:
                print("  - No se pudo normalizar la pregunta")
                return "No pude procesar tu pregunta. ¿Podrías intentar con otras palabras?"

            print(f"  - Pregunta normalizada: '{pregunta_limpia}'")

            # Clasificar la intención (categoría) usando el modelo Naive Bayes
            try:
                etiqueta_predicha = self.model.predict([pregunta_limpia])[0]
                print(f"  - Categoría predicha: '{etiqueta_predicha}'")
            except Exception as e:
                print(f"  ✗ Error al predecir la categoría: {e}")
                return "Hubo un problema al entender la categoría de tu pregunta. Por favor, intenta de nuevo."
            
            # Filtrar respuestas de esa categoría
            respuestas_categoria = self.df[self.df["categoria"] == etiqueta_predicha].copy()
            print(f"  - Se encontraron {len(respuestas_categoria)} respuestas en la categoría")

            if respuestas_categoria.empty:
                print(f"  - No hay respuestas en la categoría '{etiqueta_predicha}'")
                return f"No encontré información específica para la categoría '{etiqueta_predicha}'. Intenta con otra pregunta."

            # Vectorizar y encontrar la mejor coincidencia
            vectorizer = self.model.named_steps["tfidfvectorizer"]
            pregunta_vec = vectorizer.transform([pregunta_limpia])
            
            # Vectorizar respuestas
            respuestas_categoria["respuesta_vec"] = respuestas_categoria["respuesta_compuesta_limpia"].apply(
                lambda x: vectorizer.transform([x]) if pd.notna(x) else None
            )
            
            # Filtrar valores nulos
            respuestas_categoria = respuestas_categoria[respuestas_categoria["respuesta_vec"].notna()]
            
            if respuestas_categoria.empty:
                print("  - No se pudo vectorizar ninguna respuesta")
                return "No pude encontrar una respuesta adecuada. ¿Podrías reformular tu pregunta?"
                
            # Calcular similitud
            print("  - Calculando similitudes...")
            respuestas_categoria["similitud"] = respuestas_categoria["respuesta_vec"].apply(
                lambda x: float(cosine_similarity(x, pregunta_vec)[0][0]) if x is not None else 0.0
            )
            
            # Ordenar por similitud
            respuestas_categoria = respuestas_categoria.sort_values("similitud", ascending=False)
            
            # Mostrar las 3 mejores coincidencias
            print("\n  Mejores coincidencias encontradas:")
            for i, (_, row) in enumerate(respuestas_categoria.head(3).iterrows(), 1):
                print(f"  {i}. Similitud: {row['similitud']*100:.1f}% - '{row.get('pregunta', '')[:80]}{'...' if len(str(row.get('pregunta', ''))) > 80 else ''}")
            
            # Obtener la mejor respuesta con al menos 60% de similitud
            mejor_idx = respuestas_categoria["similitud"].idxmax()
            mejor_similitud = respuestas_categoria.loc[mejor_idx, "similitud"]
            
            print(f"\n  Mejor coincidencia: {mejor_similitud*100:.1f}% de similitud")
            
            if mejor_similitud >= 0.6:  # 60% de similitud
                return respuestas_categoria.loc[mejor_idx, "respuesta_compuesta"]
            else:
                print(f"  ✗ La mejor coincidencia no supera el umbral del 60% ({mejor_similitud*100:.1f}%)")
                
                # Si la mejor coincidencia tiene al menos 40% de similitud, sugerir posibles respuestas
                if mejor_similitud >= 0.1:
                    posibles_respuestas = respuestas_categoria.head(3)
                    sugerencias = [
                        f"- {row['pregunta']}" 
                        for _, row in posibles_respuestas.iterrows()
                    ]
                    return (
                        "No estoy seguro de entender completamente tu pregunta. "
                        "¿Te refieres a alguna de estas opciones?\n\n" +
                        "\n".join(sugerencias[:3]) + 
                        "\n\nPor favor, reformula tu pregunta para ser más específico."
                    )
                
                return "No encontré una respuesta lo suficientemente precisa. ¿Podrías ser más específico o reformular tu pregunta?"
                
        except Exception as e:
            print(f"\n  ✗ Error al procesar la pregunta: {e}")
            import traceback
            traceback.print_exc()
            return "Lo siento, hubo un error al procesar tu pregunta. Por favor, inténtalo de nuevo."

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

