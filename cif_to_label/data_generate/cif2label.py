import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageDraw
from ase.io import read
import os


BASE_INCREMENT = 100


def compute_gray_value(atomic_number):
    return int((atomic_number / 16) * (255 - BASE_INCREMENT) + BASE_INCREMENT)


element_colors = {
    'Si': (compute_gray_value(14), 0.36875),
    'Al': (compute_gray_value(13), 0.36875),
    'O': (compute_gray_value(8), 0.228125),
    'H': (compute_gray_value(1), 0.1),
    'C': (compute_gray_value(6), 0.240625),
    'S': (compute_gray_value(16), 0.31875),
    'N': (compute_gray_value(7), 0.234375)
}

print(element_colors)


class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片标注工具")

        # Variables
        self.image_path = tk.StringVar()
        self.cif_path = tk.StringVar()
        self.processed_images = []

        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Image path
        ttk.Label(frame, text="图片路径：").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.image_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="选择图片", command=self.open_image).grid(row=0, column=2, padx=5, pady=5)

        # CIF path
        ttk.Label(frame, text=".cif 路径：").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.cif_path, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="选择.cif文件", command=self.open_cif).grid(row=1, column=2, padx=5, pady=5)

        # Buttons
        ttk.Button(frame, text="标注图片", command=self.process_image).grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="保存", command=self.save_images).grid(row=2, column=2, pady=10)

    def open_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("对应尺寸的图片文件", "*.png;*.jpg;*.jpeg;*.tiff")])
        self.image_path.set(image_path)

    def open_cif(self):
        cif_dir = filedialog.askdirectory(title="选择含有.cif文件的文件夹")
        self.cif_path.set(cif_dir)

    def process_image(self):
        cif_dir = self.cif_path.get()
        image_path = self.image_path.get()
        img = Image.open(image_path)
        width, height = img.size
        cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]

        for cif_file in cif_files:
            full_cif_path = os.path.join(cif_dir, cif_file)
            try:
                atoms = read(full_cif_path)
                structure = read(full_cif_path)

                a = structure.cell[0, 0]
                c = structure.cell[2, 2]
                scale_xz = height / a
                print(scale_xz)

                img = Image.new('L', (width, height), color='black')
                draw = ImageDraw.Draw(img)

                for atom in atoms:
                    element = atom.symbol  # Get the element type (e.g., 'H', 'Si', etc.)
                    if element in element_colors:  # Check if the element type exists in the dictionary
                        position = atom.position
                        position[1] = 0  # You set the y-coordinate to 0; consider if this is needed for your use case.
                        z, y, x = atom.position

                        # Use the element type to get the appropriate color and radius
                        gray_value, radius = element_colors[element]

                        # Calculate the screen coordinates
                        x_pixel, z_pixel = int(x * scale_xz), int(z * scale_xz)

                        # Convert the radius based on your scaling (this may need adjustment based on your exact needs)
                        radius_pixel = int(radius * 6.00)

                        # Draw the atom
                        draw.ellipse((x_pixel - radius_pixel, z_pixel - radius_pixel,
                                      x_pixel + radius_pixel, z_pixel + radius_pixel),
                                     fill=gray_value)

                # Add the modified image to your processed list
                self.processed_images.append((cif_file, img))

            except FileNotFoundError:
                print(f"指定的文件 {full_cif_path} 未找到，请检查路径是否正确。")
            except Exception as e:
                print(f"处理 {cif_file} 时发生错误：{e}")

        messagebox.showinfo("完成", "所有图片处理完成！")

    def save_images(self):
        for cif_file, img in self.processed_images:
            output_name = cif_file.split('.tiff')[0] + '.png'
            output_path = os.path.join('D:\\project\\phase_structure\\phase_structure_recognition\\cif_to_label\\new images\\xinzeng', output_name)
            img.save(output_path)
        messagebox.showinfo("完成", f"已保存 {len(self.processed_images)} 张图片")


if __name__ == "__main__":
    window = tk.Tk()
    app = ImageLabelingApp(window)
    window.mainloop()
