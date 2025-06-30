# Setup

### Installing uv
Ich würde gerne mit dem Packagemanager [uv](https://docs.astral.sh/uv/) arbeiten, damit wir unsere python depentencies und package Struktur besser verwalten können.  
Die Instalation geht ganz einfach mit pi:
```bash
pip install uv
```

Um die uv environment zu erstellen müsst ihr einmalig den folgenden Befehl ausführen:
```bash
source .venv/bin/activate
```

### Using uv
Hier eine Kurse Übersicht der wichtigsten Befehle:
- ```uv sync```: Sync the project's dependencies with the environment. (Bitte immer als erstes nach dem pull ausführen, damit wir alle auf dem selben Stand sind)
- ```uv add```: Add a dependency to the project.
- ```uv remove```: Remove a dependency from the project.

### Add new package 
Unsere packages können wir einfach dem package Ordner hinzufügen und mit dem folgenden Befehl zu einem uv package machen:
```bash
uv pip install -e /path/to/name-of-package
```
Dabei ist wichtig, dass euer package-Name nicht mit Unterstrich (_), sondern Bindestrich (-) geschrieben wird, damit uv das unterschieden kann. Ab dann können wir alle unsere packages einfach mit "name_of_package" in unsere Dateien importieren.
