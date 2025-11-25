import cv2
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# ---------------------------------
# 1) Preprocessing Function (Updated with CLAHE)
# ---------------------------------
def preprocess_image(path, target_size=(256, 256)):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image {path}")
        return None

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Step 3: Denoising (Gaussian Blur)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)

    # Step 4: Contrast Enhancement (CLAHE instead of Log Transform)
    # This prevents the image from becoming too white/washed out
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(denoised)

    # Step 5: Sharpening
    # We apply sharpening on the CLAHE enhanced image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel)

    # Return all stages (Mainly we need 'sharpened' at index 4)
    return img, resized, denoised, clahe_img, sharpened


# ---------------------------------
# 2) Display the results (Updated titles)
# ---------------------------------
def show_steps(path):
    results = preprocess_image(path)
    if results is None: return

    original, gray_resized, denoised, clahe_img, sharpened = results

    titles = ["Original", "Grayscale + Resize", "Denoised", "Contrast (CLAHE)", "Sharpened"]
    images = [original, gray_resized, denoised, clahe_img, sharpened]

    plt.figure(figsize=(12, 10))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        if i == 0:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================
# 3) Run on Full Dataset (Batch Processing)
# ==========================================
def process_and_save_dataset(input_root, output_root):
    
    # Define dataset categories (Folder names)
    categories = ['Normal', 'Viral Pneumonia', 'Lung_Opacity'] 

    print(f"Starting processing from: {input_root}")
    print(f"Saving results to: {output_root}")

    for category in categories:
        
        current_input_dir = os.path.join(input_root, category)
        current_output_dir = os.path.join(output_root, category)
        
        # Check if input folder exists
        if not os.path.exists(current_input_dir):
            print(f"Skipping {category} (Folder not found)")
            continue

        # Create output directory
        os.makedirs(current_output_dir, exist_ok=True)

        # Get all images
        images_list = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images_list.extend(glob(os.path.join(current_input_dir, ext)))

        print(f"Processing {category}: {len(images_list)} images found...")

        # Loop through images
        for img_path in images_list:
            try:
                results = preprocess_image(img_path)
                if results is None: continue
                
                # Retrieve the final sharpened image (Index 4)
                final_enhanced_image = results[4] 
              
                # Save to new location
                file_name = os.path.basename(img_path)
                save_path = os.path.join(current_output_dir, file_name)
                
                cv2.imwrite(save_path, final_enhanced_image)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("Done! Enhancement complete for all images.")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Path to your cleaned dataset
    dataset_path = r"C:\DataSet\Lung X-Ray Image\Lung X-Ray Image"
    
    # Output path
    output_path = dataset_path + "_Enhanced"

    # Start processing
    process_and_save_dataset(dataset_path, output_path)