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

In Windows:

```powershell
.venv\Scripts\activate
```

### Using uv

Hier eine Kurse Übersicht der wichtigsten Befehle:

- `uv sync`: Sync the project's dependencies with the environment. (Bitte immer als erstes nach dem pull ausführen, damit wir alle auf dem selben Stand sind)
- `uv add`: Add a dependency to the project.
- `uv remove`: Remove a dependency from the project.

### Add new package

Package erstellen
```bash
uv init --package name-of-package
```

Unsere packages können wir einfach dem package Ordner hinzufügen und mit dem folgenden Befehl zu einem uv package machen:

```bash
uv pip install -e /path/to/name-of-package
```

Dabei ist wichtig, dass euer package-Name nicht mit Unterstrich (\_), sondern Bindestrich (-) geschrieben wird, damit uv das unterschieden kann. Ab dann können wir alle unsere packages einfach mit "name_of_package" in unsere Dateien importieren.


From scratch training (Baseline):

```bash
python test2.py --mode train --env MiniGrid-Crossing-5x5-v0 
```

Fine Tuning (model_path is required):
```bash
python test2.py --mode finetune --env MiniGrid-Crossing-5x5-v0 --model_path trained_models/dqn_5x5_cnn_interval__40000_steps
```

Fine Tuning sweep (the sweep list is hardcoded in test2 those are the models):
```bash
python test2.py --mode finetune_sweep --env MiniGrid-Crossing-5x5-v0
```