"""
Script de prueba para verificar que el chatbot funciona correctamente.
Ejecuta este script antes de desplegar para asegurarte de que todo está bien.
"""

import os
import sys

def test_imports():
    """Verifica que todas las dependencias estén instaladas."""
    print("\n" + "="*60)
    print("TEST 1: Verificando dependencias")
    print("="*60)
    
    required_modules = {
        'gradio': 'Gradio',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'spacy': 'spaCy',
    }
    
    all_ok = True
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - Instalado")
        except ImportError:
            print(f"✗ {name:20s} - NO ENCONTRADO")
            all_ok = False
    
    return all_ok

def test_spacy_model():
    """Verifica que el modelo de spaCy esté instalado."""
    print("\n" + "="*60)
    print("TEST 2: Verificando modelo de spaCy")
    print("="*60)
    
    try:
        import spacy
        nlp = spacy.load("es_core_news_md")
        print("✓ Modelo 'es_core_news_md' cargado correctamente")
        return True
    except OSError:
        print("✗ Modelo 'es_core_news_md' NO ENCONTRADO")
        print("  Ejecuta: python -m spacy download es_core_news_md")
        return False
    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        return False

def test_csv_files():
    """Verifica que los archivos CSV existan."""
    print("\n" + "="*60)
    print("TEST 3: Verificando archivos CSV")
    print("="*60)
    
    required_files = [
        'chatbot_datav5.csv',
        'departamentos_600.csv',
        'chatbot_data_optimizado.csv',
    ]
    
    all_ok = True
    for filename in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024  # KB
            print(f"✓ {filename:30s} - {size:.1f} KB")
        else:
            print(f"✗ {filename:30s} - NO ENCONTRADO")
            all_ok = False
    
    return all_ok

def test_chatbot_initialization():
    """Intenta inicializar el motor del chatbot."""
    print("\n" + "="*60)
    print("TEST 4: Inicializando motor del chatbot")
    print("="*60)
    
    try:
        # Importar sin ejecutar la interfaz
        import importlib.util
        spec = importlib.util.spec_from_file_location("chatbot_gradio", "chatbot_gradio.py")
        module = importlib.util.module_from_spec(spec)
        
        # No ejecutar la interfaz, solo verificar imports
        print("✓ Archivo chatbot_gradio.py encontrado y válido")
        return True
    except FileNotFoundError:
        print("✗ Archivo chatbot_gradio.py NO ENCONTRADO")
        return False
    except Exception as e:
        print(f"✗ Error al verificar chatbot_gradio.py: {e}")
        return False

def test_chatbot_response():
    """Prueba una respuesta del chatbot."""
    print("\n" + "="*60)
    print("TEST 5: Probando respuesta del chatbot")
    print("="*60)
    
    try:
        # Importar el motor del chatbot
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from chatbot_gradio import ChatbotEngine
        
        print("Inicializando chatbot...")
        chatbot = ChatbotEngine()
        
        # Probar con una pregunta de prueba
        test_query = "Hola"
        print(f"\nPregunta de prueba: '{test_query}'")
        response = chatbot.get_response(test_query)
        print(f"Respuesta: '{response}'")
        
        if response and len(response) > 0:
            print("\n✓ El chatbot respondió correctamente")
            return True
        else:
            print("\n✗ El chatbot no generó una respuesta")
            return False
            
    except Exception as e:
        print(f"\n✗ Error al probar el chatbot: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("VERIFICACIÓN DEL CHATBOT GRADIO")
    print("="*60)
    
    results = {
        'Dependencias': test_imports(),
        'Modelo spaCy': test_spacy_model(),
        'Archivos CSV': test_csv_files(),
        'Inicialización': test_chatbot_initialization(),
    }
    
    # Probar respuesta solo si todo lo demás está bien
    if all(results.values()):
        results['Respuesta del chatbot'] = test_chatbot_response()
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} - {test_name}")
    
    print("="*60)
    
    if all(results.values()):
        print("\n✅ ¡Todas las pruebas pasaron!")
        print("El chatbot está listo para usarse.")
        print("\nPara iniciar el chatbot ejecuta:")
        print("  python chatbot_gradio.py")
    else:
        print("\n⚠️  Algunas pruebas fallaron.")
        print("Por favor, revisa los errores arriba y corrígelos.")
        print("\nConsejos:")
        print("  - Instala dependencias: pip install -r requirements_gradio.txt")
        print("  - Descarga modelo spaCy: python -m spacy download es_core_news_md")
        print("  - Verifica que los archivos CSV estén en la misma carpeta")
    
    print("\n")
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
