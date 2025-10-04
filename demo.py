import tkinter as tk
from tkinter import scrolledtext, messagebox
import pandas as pd

class ChatbotApp:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot MVP")
        master.geometry("700x500")

        self.df = self.load_data("chatbot_datav5.csv")
        if self.df is None:
            master.destroy()
            return
        # Configuración de la interfaz de usuario
        self.chat_history_frame = tk.Frame(master)
        self.chat_history_frame.pack(
            padx=10, pady=10, fill=tk.BOTH, expand=True
        )

        self.chat_history = scrolledtext.ScrolledText(
            self.chat_history_frame, wrap=tk.WORD, state='disabled', font=("Arial", 10)
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        self.user_input_frame = tk.Frame(master)
        self.user_input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.user_entry = tk.Entry(self.user_input_frame, font=("Arial", 10))
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_entry.bind("<Return>", self.send_message_event)

        self.send_button = tk.Button(self.user_input_frame, text="Enviar", command=self.send_message, font=("Arial", 10))
        self.send_button.pack(side=tk.RIGHT, padx=5)

        self.display_message("Chatbot: ¡Hola! Soy tu asistente de análisis vegetal. ¿En qué puedo ayudarte hoy?", "chatbot")

    def load_data(self, filename):
        try:
            df = pd.read_csv(filename)
            return df
        except FileNotFoundError:
            messagebox.showerror("Error", f"El archivo {filename} no se encontró. Asegúrate de que esté en el mismo directorio que el script.")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos: {e}")
            return None

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

    def get_chatbot_response(self, query):
        query_lower = query.lower()
        
        # Buscar coincidencias en preguntas y respuestas compuestas
        matches = self.df[
            self.df["pregunta"].str.lower().str.contains(query_lower) |
            self.df["respuesta_compuesta"].str.lower().str.contains(query_lower)
        ]

        if not matches.empty:
            # Priorizar respuestas compuestas si están disponibles y son relevantes
            for index, row in matches.iterrows():
                if query_lower in row["pregunta"].lower() or query_lower in row["respuesta_compuesta"].lower():
                    return row["respuesta_compuesta"]
            # Si no hay una respuesta compuesta directa, tomar la primera respuesta simple
            return matches.iloc[0]["respuesta"]
        else:
            # Respuestas predefinidas para no-coincidencias
            if "hola" in query_lower or "saludo" in query_lower:
                return "¡Hola! ¿Cómo puedo ayudarte con tus análisis vegetales?"
            elif "gracias" in query_lower:
                return "De nada. Estoy aquí para ayudarte."
            elif "cultivo" in query_lower:
                return "Puedo darte información sobre cultivos, su estado, salud y niveles de macronutrientes. ¿Qué cultivo te interesa?"
            elif "nutrientes" in query_lower or "macronutrientes" in query_lower:
                return "Tengo datos sobre niveles de Nitrógeno, Fósforo, Potasio, Calcio, Magnesio, Sodio y Azufre. ¿Cuál te gustaría consultar?"
            elif "estado" in query_lower or "salud" in query_lower:
                return "Puedo informarte sobre el estado general y la salud foliar de las plantas. ¿Qué planta te interesa?"
            else:
                return "Lo siento, no encontré una respuesta específica para tu consulta. Por favor, intenta reformular tu pregunta o sé más específico."

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
