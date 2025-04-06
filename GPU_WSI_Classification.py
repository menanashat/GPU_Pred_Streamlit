import os
import cv2
import torch
import cupy as cp
import numpy as np
import streamlit as st
import openslide
import pandas as pd
from PIL import Image
from torchvision import transforms
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from collections import Counter
from torchvision.models import convnext_tiny
from PIL import ImageDraw

# -------------------------
# CONFIGURATION
# -------------------------
UPLOAD_DIR = "./uploaded_files"
OUTPUT_DIR = "./tiles_output"
RESULTS_CSV = "classification_results.csv"
TILE_SIZE = 256
TISSUE_THRESHOLD = 0.5  # Minimum tissue content per tile
MODEL_PATH = "ConvNeXt_best_model.pth"  # Replace with actual model path

# Class Names
CLASS_NAMES = [
    "Tumor Cells", 
    "Mitosis", 
    "Karyorrhexis", 
    "Stroma"
]


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    model = convnext_tiny(num_classes=20)  # Keep original num_classes for loading weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Load the state dictionary (except for classifier head)
    state_dict = {k: v for k, v in state_dict.items() if "classifier.2" not in k}
    model.load_state_dict(state_dict, strict=False)  # Load non-head weights

    # Replace classifier head with a new one matching 4 classes
    model.classifier[2] = torch.nn.Linear(in_features=768, out_features=4)
    
    model.to(device)
    model.eval()
    return model

model = load_model()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# DETECT TISSUE REGIONS
# -------------------------
def detect_tissue_regions(slide):
    thumbnail = slide.get_thumbnail((1024, 1024))
    thumbnail_gray = cp.array(cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY))

    threshold = threshold_otsu(cp.asnumpy(thumbnail_gray))
    binary_mask = thumbnail_gray < threshold
    binary_mask = closing(cp.asnumpy(binary_mask), square(5))
    labeled_mask = label(binary_mask)

    return labeled_mask

# -------------------------
# EXTRACT TILES
# -------------------------
def extract_tiles(slide, labeled_mask, downsample_factor, output_subdir):
    tiles_extracted = []
    slide_w, slide_h = slide.dimensions
    
    for region in regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        min_x, min_y = int(minc * downsample_factor), int(minr * downsample_factor)
        max_x, max_y = int(maxc * downsample_factor), int(maxr * downsample_factor)

        for x in range(min_x, max_x, TILE_SIZE):
            for y in range(min_y, max_y, TILE_SIZE):
                if x + TILE_SIZE > slide_w or y + TILE_SIZE > slide_h:
                    continue

                tile = slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert("RGB")
                tile_np = np.array(tile)
                gray_tile = cp.array(cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY))
                tissue_ratio = cp.sum(gray_tile < threshold_otsu(cp.asnumpy(gray_tile))) / (TILE_SIZE * TILE_SIZE)
                
                if tissue_ratio > TISSUE_THRESHOLD:
                    tile_path = os.path.join(output_subdir, f"tile_{x}_{y}.jpg")
                    tile.save(tile_path)
                    tiles_extracted.append(tile_path)
    
    return tiles_extracted

# -------------------------
# PROCESS SVS FILE
# -------------------------
def process_svs_file(svs_path):
    filename = os.path.basename(svs_path).replace(".svs", "")
    output_subdir = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(output_subdir, exist_ok=True)
    
    slide = openslide.OpenSlide(svs_path)
    labeled_mask = detect_tissue_regions(slide)
    
    thumb_w, thumb_h = 1024, 1024
    slide_w, slide_h = slide.dimensions
    downsample_factor = slide_w / thumb_w
    
    tile_paths = extract_tiles(slide, labeled_mask, downsample_factor, output_subdir)
    
    return tile_paths, output_subdir,downsample_factor  

# -------------------------
# PREDICT TILE CLASS
# -------------------------
def predict_tile(model, tile_path):
    img = Image.open(tile_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
    
    return CLASS_NAMES[predicted_index]

# -------------------------
# CLASSIFY TILES
# -------------------------
def classify_tiles(model, tile_paths):
    """
    Classifies each tile and stores the prediction along with softmax probabilities.
    """
    tile_predictions = {}
    tile_distributions = {}

    for tile_path in tile_paths:
        img = Image.open(tile_path)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()  # Softmax for percentages
            predicted_index = torch.argmax(output, dim=1).item()
        
        # Save predicted class and full distribution
        tile_predictions[tile_path] = CLASS_NAMES[predicted_index]
        tile_distributions[tile_path] = {CLASS_NAMES[i]: round(probabilities[i] * 100, 2) for i in range(len(CLASS_NAMES))}

    return tile_predictions, tile_distributions


# -------------------------
# AGGREGATE TILE CLASSIFICATION
# -------------------------
def aggregate_predictions(tile_predictions):
    class_counts = Counter(tile_predictions.values())
    total_tiles = sum(class_counts.values())

    class_percentages = {cls: round((count / total_tiles) * 100, 2) for cls, count in class_counts.items()}
    most_common_class = class_counts.most_common(1)[0][0]

    return most_common_class, class_percentages

# -------------------------
# SAVE RESULTS TO CSV
# -------------------------
def save_results_to_csv(tile_predictions, tile_distributions, slide_name):
    """
    Saves tile-level predictions, computes whole-slide classification, 
    and adds 'Final Classification' column.
    """
    results = []
    
    # Define key cell types
    selected_classes = ["Tumor Cells", "Mitosis", "Karyorrhexis", "Stroma"]
    
    # WSI-level accumulators
    wsi_sums = {cls: 0 for cls in selected_classes}  # Sum of percentages
    wsi_counts = {cls: 0 for cls in selected_classes}  # Count of tiles with this cell type
    wsi_cell_counts = {cls: 0 for cls in selected_classes}  # Total cell count
    total_tiles = len(tile_predictions)

    for tile_path, predicted_class in tile_predictions.items():
        row = {
            "Tile Name": tile_path,
            "Predicted Class": predicted_class,
            "Slide Name": slide_name
        }
        
        # Store both percentage and estimated cell count
        filtered_distributions = {}
        for cls in selected_classes:
            percentage = tile_distributions[tile_path].get(cls, 0)
            estimated_cell_count = int(round(percentage * TILE_SIZE * TILE_SIZE / 10000))  # Approximate count

            filtered_distributions[f"{cls} (%)"] = percentage
            filtered_distributions[f"{cls} Count"] = estimated_cell_count

            # WSI-level accumulations
            wsi_sums[cls] += percentage
            wsi_cell_counts[cls] += estimated_cell_count
            if percentage > 0:
                wsi_counts[cls] += 1  # Count tiles with this cell type

        row.update(filtered_distributions)

        # Determine **Tile-Level Final Classification**
        is_tumor_positive = filtered_distributions["Tumor Cells (%)"] > 0
        row["Final Classification"] = "Tumor Positive" if is_tumor_positive else "Tumor Negative"

        results.append(row)

    # Compute WSI-wide percentages
    wsi_percentages = {cls: round(wsi_sums[cls] / total_tiles, 2) for cls in selected_classes}

    # Determine **WSI-Level Final Classification**
    wsi_tumor_positive = wsi_percentages["Tumor Cells"] > 0
    wsi_final_classification = "Tumor Positive" if wsi_tumor_positive else "Tumor Negative"

    # Add WSI-wide summary row
    wsi_summary = {
        "Tile Name": "Overall WSI Summary",
        "Predicted Class": wsi_final_classification,
        "Slide Name": slide_name,
        "Final Classification": wsi_final_classification
    }

    # Include both percentage and absolute count in the summary
    for cls in selected_classes:
        wsi_summary[f"{cls} (%)"] = wsi_percentages[cls]
        wsi_summary[f"{cls} Count"] = wsi_cell_counts[cls]

    results.append(wsi_summary)

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    
    print("✅ Results saved successfully with 'Final Classification'!")




# -------------------------
# Overlays bounding boxes on the WSI thumbnail where tumor cells are detected.
# -------------------------


def mark_tumor_regions(slide, tumor_tiles, downsample_factor):
    try:
        # Ensure slide is correctly opened
        if slide is None:
            raise ValueError("SVS slide file is None. Cannot process.")

        # Convert slide to a thumbnail for visualization
        thumbnail = slide.get_thumbnail((1024, 1024))
        draw = ImageDraw.Draw(thumbnail)

        # Loop through tumor tiles and mark them in **red**
        for tile_path in tumor_tiles:
            try:
                # Extract X, Y from tile filename
                parts = os.path.basename(tile_path).replace("tile_", "").replace(".jpg", "").split("_")
                if len(parts) != 2:
                    print(f"⚠️ Skipping invalid tile filename: {tile_path}")
                    continue  # Skip invalid filenames

                x, y = map(int, parts)
                x_thumb, y_thumb = int(x / downsample_factor), int(y / downsample_factor)

                # Draw a **red box** around detected tumor regions
                box_size = int(256 / downsample_factor)
                draw.rectangle([x_thumb, y_thumb, x_thumb + box_size, y_thumb + box_size], outline="red", width=2)

            except Exception as e:
                print(f"⚠️ Error processing tile {tile_path}: {e}")

        return thumbnail

    except Exception as e:
        print(f"❌ Error inside mark_tumor_regions(): {e}")
        return None











# -------------------------
# mark_all_cell_types
# -------------------------



from PIL import ImageDraw, ImageFont

def mark_all_cell_types(slide, tile_predictions, downsample_factor):
    try:
        if slide is None:
            raise ValueError("SVS slide file is None. Cannot process.")

        # Convert slide to a thumbnail for visualization
        thumbnail = slide.get_thumbnail((1024, 1024))
        draw = ImageDraw.Draw(thumbnail)

        # Define color mapping for different cell types
        cell_colors = {

            "Mitosis": "blue",
            "Karyorrhexis": "purple",
            "Stroma": "green",
            "Inflammatory Cells": "orange",
            "Blood Vessels": "cyan",
            "Fibroblasts": "pink",
            "Macrophages": "yellow"
        }

        # Loop through tiles and mark locations
        for tile_path, cell_type in tile_predictions.items():
            try:
                # Extract X, Y from tile filename
                parts = os.path.basename(tile_path).replace("tile_", "").replace(".jpg", "").split("_")
                if len(parts) != 2:
                    print(f"⚠️ Skipping invalid tile filename: {tile_path}")
                    continue

                x, y = map(int, parts)
                x_thumb, y_thumb = int(x / downsample_factor), int(y / downsample_factor)

                # Select color based on cell type
                box_color = cell_colors.get(cell_type, "gray")

                # Draw box for the cell type
                box_size = int(256 / downsample_factor)
                draw.rectangle([x_thumb, y_thumb, x_thumb + box_size, y_thumb + box_size], outline=box_color, width=2)

            except Exception as e:
                print(f"⚠️ Error processing tile {tile_path}: {e}")

        return thumbnail

    except Exception as e:
        print(f"❌ Error inside mark_all_cell_types(): {e}")
        return None


# -------------------------
# STREAMLIT UI
# -------------------------

# Initialize session state variables to prevent AttributeError
if "class_distribution" not in st.session_state or st.session_state.class_distribution is None:
    st.session_state.class_distribution = {}  # Initialize as an empty dictionary

if "tile_predictions" not in st.session_state:
    st.session_state.tile_predictions = []

if "tumor_tiles" not in st.session_state:
    st.session_state.tumor_tiles = []

if "svs_path" not in st.session_state:
    st.session_state.svs_path = None

if "downsample_factor" not in st.session_state:
    st.session_state.downsample_factor = 1  # Set default downsample factor


st.title("🧪 Multi-Class Tumor Classification in Whole Slide Images (WSI) 🧪")
st.write("Upload an **SVS** file to classify the extracted tissue tiles.")

# Initialize session state variables
session_vars = [
    "svs_path", "tile_predictions", "tile_distributions", "class_distribution", 
    "tumor_tiles", "mitosis_tiles", "karyorrhexis_tiles", "marked_thumbnail", "downsample_factor"
]
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = None

# Replace the original upload block with this
uploaded_file = st.file_uploader("Upload an SVS file", type=["svs"])

if uploaded_file is not None:
    svs_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    # Reset session state if it's a new file
    if st.session_state.svs_path != svs_path:
        st.session_state.svs_path = None
        st.session_state.tile_predictions = None
        st.session_state.tile_distributions = None
        st.session_state.class_distribution = {}
        st.session_state.tumor_tiles = []
        st.session_state.mitosis_tiles = []
        st.session_state.karyorrhexis_tiles = []

    if st.session_state.tile_predictions is None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(svs_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.svs_path = svs_path
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

        st.write("🖼 Extracting tissue regions and tiling...")
        tile_paths, tile_folder, downsample_factor = process_svs_file(svs_path)
        st.session_state.downsample_factor = downsample_factor

        if not tile_paths:
            st.error("❌ No valid tissue-containing tiles found! Try another slide.")
        else:
            st.write(f"✅ Extracted **{len(tile_paths)}** tiles.")
            st.write("🔍 Running classification on extracted tiles...")
            tile_predictions, tile_distributions = classify_tiles(model, tile_paths)
            final_prediction, class_distribution = aggregate_predictions(tile_predictions)

            st.session_state.tile_predictions = tile_predictions
            st.session_state.tile_distributions = tile_distributions
            st.session_state.class_distribution = class_distribution
            st.session_state.tumor_tiles = [tile for tile, pred in tile_predictions.items() if pred == "Tumor Cells"]
            st.session_state.mitosis_tiles = [tile for tile, pred in tile_predictions.items() if pred == "Mitosis"]
            st.session_state.karyorrhexis_tiles = [tile for tile, pred in tile_predictions.items() if pred == "Karyorrhexis"]
            save_results_to_csv(tile_predictions, tile_distributions, uploaded_file.name)

# -------------------------
# DISPLAY CLASSIFICATION RESULTS
# -------------------------
if st.session_state.tile_predictions and st.session_state.class_distribution:

    # 📊 **Tile Class Distribution**
    st.write("### 📊 **Tile Class Distribution**")

    # Compute cell counts safely
    selected_classes = ["Tumor Cells", "Mitosis", "Karyorrhexis", "Stroma"]
    tile_class_counts = {
        cls: int(round(st.session_state.class_distribution.get(cls, 0) * TILE_SIZE * TILE_SIZE / 10000)) 
        for cls in selected_classes
    }

    for cls, percentage in st.session_state.class_distribution.items():
        if cls in tile_class_counts:  
            count = tile_class_counts[cls]
            st.write(f"🔹 {cls}: **{percentage}%** ({count} cells)")

    # Compute Tumor Percentage & MK Index
    tumor_percentage = st.session_state.class_distribution.get("Tumor Cells", 0)
    mk_index = (st.session_state.class_distribution.get("Mitosis", 0) +
                st.session_state.class_distribution.get("Karyorrhexis", 0))

    st.write("### 🏆 **Final Whole-Slide Classification**")

    # Determine Tumor Status
    has_tumor = tumor_percentage > 0
    slide_status = "Tumor Positive" if has_tumor else "Tumor Negative"

    st.success(f"🩸 **Predicted Slide Class:** {slide_status} (Tumor Cells: {tumor_percentage}%)")
    st.write(f"🧬 **MK Index (Mitosis + Karyorrhexis):** {mk_index}%")

    # ✅ **Download Button (No Refresh)**
    st.download_button(
        "📥 Download Classification Results",
        open(RESULTS_CSV, "rb"),
        "classification_results.csv",
        "text/csv"
    )


# Define colors for each class type (HTML Color Codes)
CLASS_COLORS = {
    "Tumor Cells": "#FF0000",      # Bright Red
    "Mitosis": "#0000FF",          # Bright Blue
    "Karyorrhexis": "#00FF00",     # Bright Green
    "Stroma": "#FFFF00",           # Yellow
    "Karyolysis": "#FFA500",       # Orange
    "Necrosis": "#800080",         # Purple
    "Inflammatory Cells": "#FFFFFF",  # White
    "Blood Vessels": "#8B0000",    # Dark Red
    "Fibroblasts": "#4682B4",      # Steel Blue
    "Macrophages": "#FF4500",      # Orange Red
    "Epithelial Cells": "#9932CC",  # Dark Orchid
    "Lymphocytes": "#DC143C",      # Crimson
    "Endothelial Cells": "#FFD700", # Gold
    "Connective Tissue": "#32CD32", # Lime Green
    "Basement Membrane": "#D2691E", # Chocolate
    "Apoptotic Bodies": "#DCDCDC",  # Gainsboro
    "Cytoplasmic Fragments": "#800080", # Purple
    "Granulocytes": "#1E90FF",     # Dodger Blue
    "Mast Cells": "#FF8C00",       # Dark Orange
    "Adipose Tissue": "#FFFF00"    # Yellow
}
# -------------------------
# CLASS DISTRIBUTION BUTTON
# -------------------------


with st.expander("📊 Class Distribution"):
    st.write("### 🔬 Full Class Breakdown")
    
    # Show all cell types except the main four
    for cls, percentage in st.session_state.class_distribution.items():
        if cls not in ["Tumor Cells", "Mitosis", "Karyorrhexis", "Stroma"]:
            color = CLASS_COLORS.get(cls, "#000000")  # Default black if not found
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{cls}: {percentage}%</span>", unsafe_allow_html=True)

    # Button to visualize all detected cell types on WSI
    if st.button("🖼 Show Class Distribution Map"):
        st.write("📍 **Visualizing Cell Type Distribution on WSI...**")

        # Ensure SVS file is available
        if st.session_state.svs_path:
            try:
                slide = openslide.OpenSlide(st.session_state.svs_path)

                # Call function to mark **all** detected cell types on the WSI
                try:
                    marked_thumbnail = mark_all_cell_types(
                        slide,
                        st.session_state.tile_predictions,
                        st.session_state.downsample_factor
                    )

                    if marked_thumbnail:
                        st.image(marked_thumbnail, caption="Class Distribution Map", use_column_width=True)
                    else:
                        st.error("❌ Failed to generate class distribution visualization.")

                except Exception as e:
                    st.error(f"❌ Error inside mark_all_cell_types(): {e}")

            except Exception as e:
                st.error(f"❌ Error opening SVS file: {e}")
        else:
            st.error("❌ SVS file not found. Please re-upload.")
      

    # -------------------------
    # TUMOR VISUALIZATION BUTTON
    # -------------------------
    if st.session_state.tumor_tiles and has_tumor and st.session_state.svs_path:

        if st.button("🔍 Check Tumor Regions"):
            st.write("🔬 Highlighting Tumor-Detected Regions...")

            # Ensure SVS file is available
            if st.session_state.svs_path:
                try:
                    slide = openslide.OpenSlide(st.session_state.svs_path)

                    # Call function to mark tumors
                    try:
                        marked_thumbnail = mark_tumor_regions(
                            slide,
                            st.session_state.tumor_tiles ,
                            st.session_state.downsample_factor
                        )

                        if marked_thumbnail:
                            st.image(marked_thumbnail, caption="Tumor Regions in WSI", use_column_width=True)
                        else:
                            st.error("❌ Failed to generate tumor region visualization.")

                    except Exception as e:
                        st.error(f"❌ Error inside mark_tumor_regions(): {e}")

                except Exception as e:
                    st.error(f"❌ Error opening SVS file: {e}")
            else:
                st.error("❌ SVS file not found. Please re-upload.")
    else:
        st.warning("⚠️ No Tumor Cells detected. Visualization disabled.")
    st.write("🗺 **Color Legend:**")
    legend_html = ""
    for cls, color in CLASS_COLORS.items():
        legend_html += f"<span style='color:{color}; font-weight:bold;'>⬤ {cls}</span> &nbsp; | &nbsp;"

    st.markdown(legend_html, unsafe_allow_html=True)      
