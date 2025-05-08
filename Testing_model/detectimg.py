import os
import cv2
import numpy as np
import pyttsx3  # Text-to-Speech
import threading  # For simultaneous speech
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Initialize Text-to-Speech engine ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Load the trained model ===
model = load_model("fruit_freshness_model.h5", compile=False)

# === Folder containing images to test ===
image_folder = r"D:\DOWNLOADS\BRAVE\Aachen\Only\Working\test_fruits"

# === Supported image extensions ===
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# === Loop through all images in the folder ===
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load {image_name}")
        continue

    # === Resize and preprocess image ===
    img = cv2.resize(image, (224, 224))
    img = img_to_array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # === Predict freshness ===
    prediction = model.predict(img, verbose=0)[0][0]

    # === Determine label ===
    label = "Fresh 🍏" if prediction < 0.5 else "Rotten 🍌"

    # === Terminal output ===
    print(f"{image_name}: {'✅ Fresh' if label == 'Fresh 🍏' else '❌ Spoiled'}")

    # === Display the image with prediction label ===
    color = (0, 255, 0) if label == "Fresh 🍏" else (0, 0, 255)
    cv2.putText(image, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fruit Freshness Prediction", image)

    # === Speak the result simultaneously ===
    voice_output = f"The fruit is {'fresh' if label == 'Fresh 🍏' else 'rotten'}"
    threading.Thread(target=lambda: [engine.say(voice_output), engine.runAndWait()]).start()

    # Wait a few seconds then move on
    if cv2.waitKey(30000) & 0xFF == ord('q'):  # Wait 3 seconds or press 'q' to quit early
        break

cv2.destroyAllWindows()
