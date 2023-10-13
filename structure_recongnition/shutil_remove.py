import os


def remove_prefix(folder_path, prefix):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file_name in files:
        if file_name.startswith(prefix):
            # Remove the prefix from the file name
            new_file_name = file_name[len(prefix):]
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_file_name)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")

# Path to the folder containing files to rename
folder_path = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\test'

# Prefix to remove from file names
prefix = 'SR_reconstructed_'

# Call the function to remove the prefix from file names
remove_prefix(folder_path, prefix)
