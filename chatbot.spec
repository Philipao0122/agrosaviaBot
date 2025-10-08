# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

def get_spacy_model_path(model_name):
    import spacy.util
    import os
    model_path = spacy.util.get_package_path(model_name)
    if not os.path.exists(model_path):
        return None
    return (os.path.dirname(model_path), os.path.join('spacy', model_name))

# Obtener rutas de los modelos de spaCy
spacy_models = []
for model in ['es_core_news_md', 'en_core_web_sm']:
    path = get_spacy_model_path(model)
    if path:
        spacy_models.append(path)

# Agregar modelos de spaCy a los datos
datas = [
    ('chatbot_datav5.csv', '.'),
    ('ubicacion_cultivos.csv', '.'),
    ('departamentos_600.csv', '.'),
] + spacy_models

a = Analysis(
    ['chatbot_tkinter_mvp.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'sklearn',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        'scipy.sparse.csgraph._validation',
        'scipy.spatial.transform._rotation_groups',
        'scipy.special.cython_special',
        'pandas',
        'numpy',
        'spacy',
        'es_core_news_md',
        'en_core_web_sm',
        'blis',
        'thinc',
        'thinc.extra.search',
        'thinc.linalg',
        'thinc.neural.ops',
        'thinc.neural.ops.blis',
        'thinc.backends.cblas',
        'thinc.backends.linalg',
        'thinc.backends.numpy_ops',
        'thinc.backends.ops',
        'thinc.backends.cpu',
        'thinc.backends.murmurhash',
        'thinc.backends.murmurhash.mrmr',
        'thinc.backends.murmurhash.about',
        'thinc.backends.murmurhash.imports',
        'srsly.msgpack.util',
        'srsly.ujson',
        'srsly.msgpack',
        'catalogue',
        'preshed',
        'cymem',
        'murmurhash',
        'tqdm',
        'tqdm.auto',
        'tqdm.std',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Asegurarse de incluir las DLLs necesarias
for d in a.binaries:
    if 'mkl' in d[0].lower():
        a.binaries.append((d[0], d[1], 'BINARY'))
    if 'libiomp' in d[0].lower():
        a.binaries.append((d[0], d[1], 'BINARY'))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ChatbotAgrosavia',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)
