from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# Charger un modèle YOLO pré-entrainé
model = YOLO("runs/detect/train13/weights/epoch15.pt")  # Mettre le chemin vers vos poids

# Fonction pour prédire sur une image et sauvegarder le résultat
def predict_and_save(image_path, output_dir):
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    # Prédire avec le modèle
    results = model.predict(source=image_path, conf=0.3)
    detections = results[0].boxes  # Obtenir les boîtes détectées
    
    # Dessiner les détections sur l'image
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = model.names[int(box.cls[0])]  # Nom de la classe prédite
        confidence = box.conf[0]  # Confiance
        color = (0, 255, 0)  # Couleur du rectangle
        thickness = 2

        # Dessiner le rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Ajouter le label et la confiance
        text = f"{label} {confidence:.2f}"
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x, text_y = x1, y1 - 10
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y), color, -1)
        cv2.putText(img, text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder l'image annotée
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"pred_{base_name}")
    cv2.imwrite(output_path, img)
    print(f"Image sauvegardée avec les prédictions : {output_path}")

if __name__ == "__main__":
    import argparse

    # Ajouter les arguments CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Chemin vers le répertoire contenant les images")
    parser.add_argument("--output-dir", default="predictions", help="Répertoire de sortie pour les images annotées")
    args = parser.parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir

    # Vérifier que le répertoire d'images existe
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Le répertoire {image_dir} n'existe pas.")

    # Traiter toutes les images dans le répertoire
    for image_file in tqdm(os.listdir(image_dir), desc="Traitement des images"):
        if image_file.endswith(".jpg"):  # Filtrer les fichiers JPEG
            image_path = os.path.join(image_dir, image_file)
            predict_and_save(image_path, output_dir)
