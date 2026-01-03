import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    model = fcn_resnet50(weights=None, aux_loss=True)

    # 🔴 MATCH TRAINING EXACTLY
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)

    # Aux classifier WAS NOT modified during training
    # So keep default ADE20K aux head (21 classes)
    # DO NOT TOUCH aux_classifier

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor

# Post-process Mask

def clean_wall_mask(mask):
    mask = (mask == 1).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, contours, -1, 255, cv2.FILLED)
    return clean

# Area Calculation

def calculate_area(mask, width_ft, height_ft):
    total_area = width_ft * height_ft
    paintable_ratio = np.sum(mask == 255) / mask.size
    paintable_sqft = paintable_ratio * total_area
    paintable_sqm = paintable_sqft * 0.092903
    return paintable_ratio, paintable_sqft, paintable_sqm

# MAIN PROGRAM

def main():
    print("\n🧱 WALL PAINTABLE AREA ESTIMATOR")
    print("--------------------------------")

    image_path = input("📷 Enter image path: ").strip()
    wall_width = float(input("📏 Enter wall width (ft): "))
    wall_height = float(input("📐 Enter wall height (ft): "))

    model_path = "wall_segmentation_fcn.pth"

    print("\n⏳ Loading model...")
    model = load_model(model_path)

    print("⏳ Processing image...")
    img, input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        pred_mask = output.argmax(1).cpu().numpy()[0]

    clean_mask = clean_wall_mask(pred_mask)

    ratio, area_sqft, area_sqm = calculate_area(
        clean_mask, wall_width, wall_height
    )

    # RESULTS

    print("\n✅ RESULT")
    print("--------------------------------")
    print(f"Total Wall Area     : {wall_width * wall_height:.2f} sq.ft")
    print(f"Paintable Area      : {area_sqft:.2f} sq.ft")
    print(f"Paintable Area      : {area_sqm:.2f} sq.m")
    print(f"Coverage Percentage : {ratio * 100:.2f}%")

    # VISUAL OUTPUT

    plt.figure(figsize=(14,5))

    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Wall Mask")
    plt.imshow(clean_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Paintable Area Overlay")
    plt.imshow(img.resize((128,128)))
    plt.imshow(clean_mask, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
