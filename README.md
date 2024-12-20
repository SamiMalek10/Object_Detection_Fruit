# Object Detection avec YOLOv8

## Description du Projet
Ce projet utilise le modèle préentraîné YOLOv8 pour la détection d'objets dans des images, des GIFs, et des vidéos. Les données d'entraînement proviennent du site [Open Images Dataset Visualizer](https://storage.googleapis.com/openimages/web/visualizer/index.html). L'objectif est d'entraîner un modèle performé pour détecter des classes choisies et générer des résultats précis avec des visualisations et des métriques.


## Prérequis

Installez les bibliothèques nécessaires :

```bash
pip install ultralytics cv2 tqdm pycocotools
```

---

## Utilisation des Scripts

### 1. Entraînement du Modèle
Utilisez `main.py` pour entraîner le modèle YOLOv8. Configurez le fichier `config.yaml` pour spécifier les chemins des données d'entraînement, de validation, et les noms des classes.

### 2. Prédiction sur les Images
Pour prédire les objets dans un dossier d'images :

```bash
python predict_images.py --image-dir /chemin/vers/images --output-dir /chemin/vers/dossier_résultats
```


### 3. Prédiction sur les GIFs
Pour prédire les objets dans un dossier de GIFs :

```bash
python predict_gifs.py --input-dir /chemin/vers/gifs --output-dir /chemin/vers/dossier_résultats
```

### 4. Prédiction sur les Vidéos
Pour prédire les objets dans une vidéo :

```bash
python predict_videos.py
```

### 5. Assemblage des Résultats
Pour assembler tous les résultats dans un fichier CSV unique :

```bash
python assemble_results.py
```

### 6. Affichage des Résultats
Pour afficher les résultats dans un dossier spécifié :

```bash
python affichage_results.py
```

---

## Remarques 

- Les poids du modèle entraîné sont sauvegardés dans `runs/detect/train13/weights/`.
- Les métriques et visualisations sont automatiquement générées dans le dossier `outputs/`.
- Pour ajuster les époques ou les classes, modifiez le fichier `config.yaml`.

N'hésitez pas à explorer les logs dans `runs/detect/train13/` pour déboguer ou analyser les performances du modèle.

