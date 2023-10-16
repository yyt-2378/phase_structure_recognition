from tkinter import Tk, Button, Label, Entry, filedialog, messagebox
from PIL import Image, ImageDraw
from ase.io import read

element_colors = {
    'Si': (130, 0.36875),
    'Al': (225, 0.36875),
    'O': (145, 0.228125),
    'H': (234, 0.1),
    'C': (136, 0.240625),
    'S': (246, 0.31875),
    'N': (192, 0.234375)
}

processed_images = []


def open_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tiff")])
    image_entry.delete(0, 'end')
    image_entry.insert(0, image_path)


def open_cif():
    cif_path = filedialog.askopenfilename(filetypes=[("CIF files", "*.cif")])
    cif_entry.delete(0, 'end')
    cif_entry.insert(0, cif_path)


def process_image():
    image_path = image_entry.get()
    cif_path = cif_entry.get()

    try:
        img = Image.open(image_path)
        width, height = img.size
        atoms_list = []
        atoms = read(cif_path)
        structure = read(cif_path)
        a = structure.cell[0, 0]
        c = structure.cell[2, 2]
        atoms_list.append(atoms)
        scale_xz = height / a

        img = Image.new('L', (width, height), color='black')
        draw = ImageDraw.Draw(img)

        for atom in atoms:
            position = atom.position
            position[1] = 0
            element = atom.symbol
            x, y, z = atom.position
            gray_value, radius = element_colors[element]
            x_pixel, z_pixel = int(z * scale_xz), int(x * scale_xz)
            radius_pixel = int(radius * 6.00)

            draw.ellipse((x_pixel - radius_pixel, z_pixel - radius_pixel,
                          x_pixel + radius_pixel, z_pixel + radius_pixel),
                         fill=gray_value)

        processed_images.append(img)

        messagebox.showinfo("Mission accomplished", "Image processing is complete！")

    except FileNotFoundError:
        print("The specified file cannot be found, please check if the path is correct.")
    except Exception as e:
        print(f"error：{e}")


def save_images():
    for i, img in enumerate(processed_images):
        img.save(f'output_{i}.png')
    messagebox.showinfo("completed!", f"saved {len(processed_images)} images")


window = Tk()
window.title("image labeling")

image_label = Label(window, text="image path：")
image_label.pack()
image_entry = Entry(window)
image_entry.pack()

cif_label = Label(window, text=".cif path：")
cif_label.pack()
cif_entry = Entry(window)
cif_entry.pack()

image_button = Button(window, text="choose image", command=open_image)
image_button.pack()

cif_button = Button(window, text="choose .cif file", command=open_cif)
cif_button.pack()

process_button = Button(window, text="label image", command=process_image)
process_button.pack()

save_button = Button(window, text="save", command=save_images)
save_button.pack()

window.mainloop()