##  Requisitos
Antes de comenzar, asegúrate de tener instalado:
- Python 3.10.X o superior
- pip

Verificar versión de Python:

```bash
python --version
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/carlos-cas3/FaceRecognizer-PCyP.git
cd FaceRecognizer-PCyP
```

### 2️. Crear entorno virtual

```bash
python -m venv venv
```

### 3️. Activar entorno virtual

En Windows:

```bash
venv\Scripts\activate
```

### 4️. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Configuración

Antes de ejecutar el programa, debes modificar el archivo:

```
config_simple.yaml
```

Cambiar la IP por la dirección IP de la máquina Linux:

```yaml
server_ip: "192.168.X.X"
```

---

## Ejecución

Terminado los pasos anteriores, ejecutar:

```bash
python main_simple.py
```

## Notas
- Ejecuta siempre el programa dentro del entorno virtual.
