### Dataset
Awl 7aja kat9lb 3la dataset mn Kaggle wla chi site : 7na l9inaha f wa7d site open source:
https://storage.googleapis.com/openimages/web/visualizer/index.html

kat5tar wa7d les classes libghiti lmodel dyalk it3lmn mnhom , st3mlna YOLOv8(predtrained model kidetecter lk les classes liktd5lhom lih) 7it hya la solution la plus rapide 3la rcnn faster rcnn...

config.yaml : rédirige les chemins (dataset , données d'entrainement ,données de validation , les noms des classes)

main.py : entrainement encore le modèle (ana 7bsst f train13 , st3mlt en globe 32 epochs l9it lmodele plus perfomrent o7bsst , chaque epoch dat liya 20 min)

predict_chi 7aja.py : katpredicter les objets li fdik l7aja o katcreer des dossiers fihum les résultats

assemble_resultats.py : fach salit mn les entrainement , jm3t ga3 les resultat.csv f resultat wa7d smito combined_results.csv(li mno an estimer en globs les metrics de test (apres 32 epochs) bach nvalider lmodele)

affichage_results.py : kat afficher les resultats f wa7d dossier smito output , o katzid f train13 dossier smito val fih ga3 les visualizations li kat3tihom Yolo par defaut(b7al Matrice de confusion) 

--image-dir : argument (chemin d'images li baghi tester 3lihom )
--input-dir : argument (chemin des gifs li baghi tester 3lihom )
--output-dir : argument (chemin du dossier fin baghi t7t fihom resultats des tests)
### commandes:
1. pip install ultralytics cv2 tqdm pycocotools

2. python predict_images.py --image-dir /path/to/images --input-dir /path/to/folders_of_resluts

ex : python predict_images.py --image-dir "C:\Users\HP\Downloads\Object-Detection-Yolo\Object-Detection-Yolo\Test\test" 
--output-dir "C:\Users\HP\Downloads\Object-Detection-Yolo\Object-Detection-Yolo\Test\results of test" 

3. python predict_gifs.py --input-dir /path/to/images --input-dir /path/to/folders_of_resluts
ex : python predict_gifs.py --input-dir "C:\Users\HP\Downloads\Test\test\gifs" --output-gif "C:\Users\HP\Downloads\Test\results of test\gifs"

4. python predict_video.py

5. python assemble_results.py

5. python afficher_results.py


