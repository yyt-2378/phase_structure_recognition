import os

def rename_files_in_directory(directory="."):
    """
    Rename files in the specified directory from format 'SR_reconstructed_molecule_*_r30_blur.png'
    to 'molecule_*_r30_blur.png'
    """
    for filename in os.listdir(directory):
        if filename.startswith("SR_reconstructed_molecule_"):
            new_name = filename.replace("SR_reconstructed_", "")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Renamed {filename} to {new_name}")

if __name__ == "__main__":
    rename_files_in_directory('/root/autodl-tmp/phase_structure_recognition/faster_rcnn_stem_dataset/test_1000000dose')
