@echo off
echo Limpiando compilaciones anteriores...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "ChatbotAgrosavia.spec" del /f /q "ChatbotAgrosavia.spec"

echo Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate

echo Actualizando pip...
python -m pip install --upgrade pip

echo Instalando dependencias...
pip install -r requirements.txt

:: Instalar PyInstaller
echo Instalando PyInstaller...
pip install --upgrade pyinstaller

:: Instalar modelos de spaCy
echo Instalando modelos de spaCy...
python -m spacy download es_core_news_md
python -m spacy download en_core_web_sm

:: Instalar dependencias faltantes
echo Instalando dependencias adicionales...
pip install --upgrade numpy pandas scikit-learn scipy thinc blis

:: Crear el ejecutable
echo Creando el ejecutable...
pyinstaller --clean --noconfirm chatbot.spec

:: Mover el ejecutable a la carpeta raíz
if exist "dist\ChatbotAgrosavia.exe" (
    move /Y "dist\ChatbotAgrosavia.exe" .
    echo ¡Ejecutable creado exitosamente como ChatbotAgrosavia.exe!
    echo.
    echo Para ejecutar el programa, haz doble clic en ChatbotAgrosavia.exe
) else (
    echo Error al crear el ejecutable. Verifica los mensajes de error.
)

pause
