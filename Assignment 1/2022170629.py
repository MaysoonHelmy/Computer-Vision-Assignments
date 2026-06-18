import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

imgs_dir = r"E:\7th term FCIS\Vision\Assignment 1\train\images"
masks_dir = r"E:\7th term FCIS\Vision\Assignment 1\train\masks"
test_pic = "test.jpg"

SEG_COLORS = {
    "road_surface": np.array([0, 0, 255]),
    "marking": np.array([255, 255, 0]),
    "road_sign": np.array([255, 0, 0]),
    "car": np.array([255, 165, 0]),
    "background": np.array([255, 0, 255])
}

labels_list = ["road_surface", "marking", "road_sign", "car", "background"]

# Classify Pixel to the class color
def assign_class_by_color(mask_rgb):
    H, W = mask_rgb.shape[:2]
    flat_mask = mask_rgb.reshape(-1, 3)
    mapped = np.zeros(H * W, dtype=int)

    for idx, (lbl, col) in enumerate(SEG_COLORS.items()):
        dist = np.sqrt(np.sum((flat_mask - col) ** 2, axis=1))
        if idx == 0:
            min_dist = dist
            mapped = np.zeros_like(dist, dtype=int)
        else:
            closer = dist < min_dist
            mapped[closer] = idx
            min_dist = np.minimum(min_dist, dist)

    return mapped.reshape(H, W)

pixels_by_class = {n: {"R": [], "G": [], "B": []} for n in labels_list}

# Load Training Data
img_files = [f for f in os.listdir(imgs_dir) if f.lower().endswith('.png')]

for name in img_files:
    img_path = os.path.join(imgs_dir, name)
    msk_path = os.path.join(masks_dir, name)

    img_raw = cv2.imread(img_path)
    mask_raw = cv2.imread(msk_path)

    if img_raw.shape[:2] != mask_raw.shape[:2]:
        mask_raw = cv2.resize(mask_raw, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

    rgb_img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    rgb_mask = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)

    class_id_map = assign_class_by_color(rgb_mask)

    # Extract RGB values 
    for idx, cname in enumerate(labels_list):
        class_pixels = (class_id_map == idx)
        if np.any(class_pixels):
            pixels_by_class[cname]["R"].extend(rgb_img[:, :, 0][class_pixels].flatten())
            pixels_by_class[cname]["G"].extend(rgb_img[:, :, 1][class_pixels].flatten())
            pixels_by_class[cname]["B"].extend(rgb_img[:, :, 2][class_pixels].flatten())

for cname in labels_list:
    for ch in ["R", "G", "B"]:
        pixels_by_class[cname][ch] = np.array(pixels_by_class[cname][ch], dtype=np.float32)


# Naive bayes 
naive_model = {}
total_count = sum(len(pixels_by_class[c]["R"]) for c in labels_list)

for cname in labels_list:
    count = len(pixels_by_class[cname]["R"])
    prior_val = count / total_count if total_count > 0 else 0.0

    naive_model[cname] = {
        "prior": prior_val,
        "mean": {},
        "var": {}
    }

    for ch in ["R", "G", "B"]:
        arr = pixels_by_class[cname][ch]
        if len(arr) > 0:
            mean_val = np.mean(arr)
            var_val = np.var(arr)
            var_val = max(var_val, 1.0)  

        naive_model[cname]["mean"][ch] = mean_val
        naive_model[cname]["var"][ch] = var_val


def Gaussian_Naive_bayes(x, mean, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mean) ** 2 / var)


# Test Image
test_img = cv2.imread(test_pic)
rgb_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB).astype(np.float32)
H, W = test_img.shape[:2]

post_probs = np.zeros((H, W, len(labels_list)))

for i, cname in enumerate(labels_list):
    prior_c = naive_model[cname]["prior"]
    likelihood = np.ones((H, W), dtype=np.float32)
    
    for j, ch in enumerate(["R", "G", "B"]):
        mu = naive_model[cname]["mean"][ch]
        sigma2 = naive_model[cname]["var"][ch]
        vals = rgb_test[:, :, j]
        likelihood *= Gaussian_Naive_bayes(vals, mu, sigma2)

    post_probs[:, :, i] = prior_c * likelihood

pred_class = np.argmax(post_probs, axis=2)

output_mask = np.zeros((H, W, 3), dtype=np.uint8)
road_area = (pred_class == 0)
output_mask[road_area] = [255, 255, 255]
output_mask[~road_area] = [0, 0, 0]

# PLot
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("Original Input")
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Road Area")
plt.imshow(output_mask)
plt.axis("off")

plt.tight_layout()
plt.show()

# Output
cv2.imwrite("Result.png", cv2.cvtColor(output_mask, cv2.COLOR_RGB2BGR))
