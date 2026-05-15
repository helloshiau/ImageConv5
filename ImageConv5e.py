# 2026-05-15 ImageConv v5e
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# Import Matplotlib modules for Tkinter integration
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analyzer Workstation")
        self.root.geometry("1400x950")

        # Core image and control variables
        self.cv_original = None
        self.cv_current = None
        self.tk_display = None
        self.display_ratio = 1.0
        self.img_offset = (0, 0)

        self._create_menu()
        self._setup_ui()

    def _create_menu(self):
        """Create the English menu bar"""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Reset Image", command=self.reset_image)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="Software Info",
                               command=lambda: messagebox.showinfo("About", "Image Analysis Tool v5.0 \n (CC) YC Shaw \n helloshiau@gmail.com \n 2026 v5e"))

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        menubar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menubar)

    def _setup_ui(self):
        """Configure the UI layout based on exact specifications"""
        # 1. Load Button
        tk.Button(self.root, text="Select Image File", command=self.load_image, width=20, height=2).pack(pady=10)

        # 2. Path Display Field
        self.path_entry = tk.Entry(self.root, width=150, justify='center')
        self.path_entry.pack(pady=5)

        # 3. Dual Canvas Layout (Horizontal Alignment)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10)

        # Canvas1: Image Display (Strictly 900x600)
        self.canvas1 = tk.Canvas(canvas_frame, width=900, height=600, bg="#1e1e1e")
        self.canvas1.pack(side=tk.LEFT, padx=10)
        self.canvas1.bind("<Button-1>", self.get_pixel_data)

        # Canvas2: Histogram Frame Container (Strictly 400x300)
        self.hist_frame = tk.Frame(canvas_frame, width=400, height=300)
        self.hist_frame.pack(side=tk.LEFT, padx=10)
        self.hist_frame.pack_propagate(False)  # Locks dimensions

        # Initialize Matplotlib Figure
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. Five Function Filters
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        filters = [
            ("Negative", self.apply_negative), ("Sharp", self.apply_sharp),
            ("Edge", self.apply_edge), ("X2", self.apply_x2), ("to Gray", self.apply_gray)
        ]
        for text, cmd in filters:
            tk.Button(btn_frame, text=text, width=15, height=2, command=cmd).pack(side=tk.LEFT, padx=5)

        # 5. Pixel Info Row (Aligned on the exact same line)
        info_line_frame = tk.Frame(self.root)
        info_line_frame.pack(pady=5)

        # Left static text with exact required string structure
        tk.Label(info_line_frame, text="Mouse Left Button for Pixel Data : ", font=("Arial", 11, "bold"),
                 fg="blue").pack(side=tk.LEFT)
        # Right dynamic data label
        self.pixel_info_label = tk.Label(info_line_frame, text="Click on the image to view data",
                                         font=("Courier", 12, "bold"), fg="#333333")
        self.pixel_info_label.pack(side=tk.LEFT, padx=5)

        # 6. Save Button
        tk.Button(self.root, text="Save Image to Another Directory", command=self.save_image,
                  bg="#28a745", fg="white", width=35, height=2).pack(pady=20)

    def load_image(self):
        """Load image file and initialize states"""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

            # Read through a NumPy buffer to natively support non-ASCII/Unicode paths
            raw_data = np.fromfile(path, dtype=np.uint8)
            self.cv_original = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            self.cv_current = self.cv_original.copy()
            self._update_interface()

    def _update_interface(self):
        """Synchronously refresh both image and histogram"""
        self._render_image_canvas()
        self._render_histogram()

    def _render_image_canvas(self):
        """Render OpenCV matrix onto Canvas1 (900x600)"""
        if self.cv_current is not None:
            if len(self.cv_current.shape) == 2:
                img_rgb = cv2.cvtColor(self.cv_current, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(img_rgb)
            orig_w, orig_h = pil_img.size

            # Scale proportionally to fit 900x600 boundaries
            pil_img.thumbnail((900, 600))
            new_w, new_h = pil_img.size
            self.display_ratio = orig_w / new_w

            self.tk_display = ImageTk.PhotoImage(pil_img)
            self.canvas1.delete("all")
            # Centered rendering
            self.canvas1.create_image(450, 300, anchor=tk.CENTER, image=self.tk_display)
            self.img_offset = (450 - new_w // 2, 300 - new_h // 2)

    def _render_histogram(self):
        """Calculate and plot R, G, B histogram into Canvas2 container"""
        if self.cv_current is not None:
            self.ax.clear()
            self.ax.set_xlim([0, 255])
            self.ax.get_yaxis().set_visible(False)  # Hides Y axis numbers for a clean look
            self.ax.tick_params(labelsize=8)
            self.ax.set_title("R G B Histogram", fontsize=9)

            if len(self.cv_current.shape) == 2:
                # Single channel plot for grayscale images
                hist = cv2.calcHist([self.cv_current], [0], None, [256], [0, 256])
                self.ax.plot(hist, color='gray', linewidth=1.5)
                self.ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.2)
            else:
                # Triple channel plots for BGR images
                colors = (('b', 'blue'), ('g', 'green'), ('r', 'red'))
                for i, (col_code, col_name) in enumerate(colors):
                    hist = cv2.calcHist([self.cv_current], [i], None, [256], [0, 256])
                    self.ax.plot(hist, color=col_name, linewidth=1.2, alpha=0.8)

            self.fig.tight_layout()
            self.hist_canvas.draw()

    def get_pixel_data(self, event):
        """Handle mouse click and update info string inline on the right side"""
        if self.cv_current is None: return

        # Deduct coordinate margins and apply inverse display ratio
        ix = int((event.x - self.img_offset[0]) * self.display_ratio)
        iy = int((event.y - self.img_offset[1]) * self.display_ratio)

        h, w = self.cv_current.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            pixel = self.cv_current[iy, ix]
            if len(self.cv_current.shape) == 2:  # Grayscale
                r = g = b = pixel
            else:  # BGR Matrix
                b, g, r = pixel

            # Modify target label text directly on the same row
            self.pixel_info_label.config(text=f"X: {ix}, Y: {iy} | R: {r}, G: {g}, B: {b}")

    def reset_image(self):
        if self.cv_original is not None:
            self.cv_current = self.cv_original.copy()
            self._update_interface()

    # --- Core Image Filters ---
    def apply_negative(self):
        if self.cv_current is not None:
            self.cv_current = 255 - self.cv_current
            self._update_interface()

    def apply_sharp(self):
        if self.cv_current is not None:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            self.cv_current = cv2.filter2D(self.cv_current, -1, kernel)
            self._update_interface()

    def apply_edge(self):
        if self.cv_current is not None:
            gray = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY) if len(
                self.cv_current.shape) == 3 else self.cv_current
            self.cv_current = cv2.Canny(gray, 100, 200)
            self._update_interface()

    def apply_x2(self):
        if self.cv_current is not None:
            # Upgrade matrix datatype to uint16 temporarily to safely avoid overflow wrapping
            res = self.cv_current.astype(np.uint16) * 2
            self.cv_current = np.clip(res, 0, 255).astype(np.uint8)
            self._update_interface()

    def apply_gray(self):
        if self.cv_current is not None and len(self.cv_current.shape) == 3:
            self.cv_current = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY)
            self._update_interface()

    def save_image(self):
        """Export current matrix out to a specific filesystem path"""
        if self.cv_current is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"), ("JPG Files", "*.jpg")])
            if save_path:
                ext = os.path.splitext(save_path)[1]
                _, buf = cv2.imencode(ext, self.cv_current)
                buf.tofile(save_path)
                messagebox.showinfo("Success", "Image has been saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()
