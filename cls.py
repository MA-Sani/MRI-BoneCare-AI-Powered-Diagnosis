import os

# Files to clear
files_to_clear = [
    r"C:\Users\KTECH\PycharmProjects\p1\BoneCare_AI\src\results\validation_report.txt",
    r"C:\Users\KTECH\PycharmProjects\p1\BoneCare_AI\src\results\segmentation_results.csv",
    r"C:\Users\KTECH\PycharmProjects\p1\BoneCare_AI\src\results\detection_results.csv",
    r"C:\Users\KTECH\PycharmProjects\p1\BoneCare_AI\src\results\classification_results.csv"
]

# Clear file contents
for file_path in files_to_clear:
    if os.path.exists(file_path):
        open(file_path, "w").close()

# Clear segmentation mask directory
segmentation_masks_dir = r"C:\Users\KTECH\PycharmProjects\p1\BoneCare_AI\src\results\segmentation_masks"
if os.path.exists(segmentation_masks_dir):
    for filename in os.listdir(segmentation_masks_dir):
        file_path = os.path.join(segmentation_masks_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

print("Results and masks cleared.")
