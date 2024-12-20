from ultralytics import YOLO
from PIL import Image, ImageSequence
import cv2
import os

# Load the YOLO model
model = YOLO("runs/detect/train13/weights/epoch15.pt")  # Update with your weights path
threshold = 0.45  # Confidence threshold for predictions

def predict_on_gif(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all GIF files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".gif"):  # Filter only GIF files
            input_gif_path = os.path.join(input_dir, file_name)
            output_gif_path = os.path.join(output_dir, f"predicted_{file_name}")

            print(f"Processing: {input_gif_path}")
            try:
                # Load the GIF using PIL
                gif = Image.open(input_gif_path)
                frames = []  # List to store processed frames

                # Process each frame
                for frame in ImageSequence.Iterator(gif):
                    frame_rgb = frame.convert("RGB")
                    frame_cv = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)

                    # Predict using YOLO
                    results = model(frame_cv)[0]

                    # Draw predictions
                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result
                        if score > threshold:
                            cv2.rectangle(frame_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            label = results.names[int(class_id)].upper()
                            cv2.putText(frame_cv, f"{label} {score:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # Convert back to PIL and append
                    frame_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
                    frames.append(frame_pil)

                # Save the annotated GIF
                frames[0].save(output_gif_path, save_all=True, append_images=frames[1:],
                               duration=gif.info.get('duration', 100), loop=gif.info.get('loop', 0))
                print(f"Saved: {output_gif_path}")
            except Exception as e:
                print(f"Error processing {input_gif_path}: {e}")

if __name__ == "__main__":
    import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-gif", required=True, help="Path to directory containing GIF files")
parser.add_argument("--output-gif", required=True, help="Directory to save annotated GIFs")
args = parser.parse_args()

predict_on_gif(args.input_gif, args.output_gif)








