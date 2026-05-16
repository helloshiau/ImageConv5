# 2026-05-16
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# Import Matplotlib libraries for Tkinter window nesting
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class StarAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Star Analysis Workstation")
        self.root.geometry("1400x950")

        # Core app structural variables
        self.cv_original = None
        self.cv_current = None
        self.tk_display = None
        self.display_ratio = 1.0
        self.img_offset = (0, 0)

        self._create_menu()
        self._setup_ui()

    def _create_menu(self):
        """Build app menu dropdown structures"""
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
                               command=lambda: messagebox.showinfo("About", "Star Analyzer App v8.0"))

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        menubar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menubar)

    def _setup_ui(self):
        """Generate app interface controls and layout panels"""
        # 1. Main action load button
        tk.Button(self.root, text="Select Image File", command=self.load_image, width=20, height=2).pack(pady=10)

        # 2. Workspace path text box
        self.path_entry = tk.Entry(self.root, width=150, justify='center')
        self.path_entry.pack(pady=5)

        # 3. Double-Canvas arrangement system
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10)

        # Canvas1: Core Image Display Space (900x600)
        self.canvas1 = tk.Canvas(canvas_frame, width=900, height=600, bg="#1a1a1a")
        self.canvas1.pack(side=tk.LEFT, padx=10)
        self.canvas1.bind("<Button-1>", self.get_pixel_data)

        # Canvas2: Fixed Histogram Plot Area (400x300)
        self.hist_frame = tk.Frame(canvas_frame, width=400, height=300)
        self.hist_frame.pack(side=tk.LEFT, padx=10)
        self.hist_frame.pack_propagate(False)

        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. Six Feature Action Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        filters = [
            ("Negative", self.apply_negative),
            ("Sharp", self.apply_sharp),
            ("Edge", self.apply_edge),
            ("X2", self.apply_x2),
            ("to Gray", self.apply_gray),
            ("Star Enhance", self.apply_star_enhance)
        ]
        for text, cmd in filters:
            tk.Button(btn_frame, text=text, width=14, height=2, command=cmd).pack(side=tk.LEFT, padx=4)

        # 5. Inline Position Data Row
        info_line_frame = tk.Frame(self.root)
        info_line_frame.pack(pady=5)

        tk.Label(info_line_frame, text="Mouse Left Button for Pixel Data : ", font=("Arial", 11, "bold"),
                 fg="blue").pack(side=tk.LEFT)
        self.pixel_info_label = tk.Label(info_line_frame, text="Click image to extract values",
                                         font=("Courier", 12, "bold"), fg="#222222")
        self.pixel_info_label.pack(side=tk.LEFT, padx=5)

        # 6. File exporter trigger button
        tk.Button(self.root, text="Save Image to Another Directory", command=self.save_image,
                  bg="#28a745", fg="white", width=35, height=2).pack(pady=20)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

            # Use raw buffer logic to process folders with Unicode characters correctly
            raw_data = np.fromfile(path, dtype=np.uint8)
            self.cv_original = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            self.cv_current = self.cv_original.copy()
            self._update_interface()

    def _update_interface(self):
        self._render_image_canvas()
        self._render_histogram()

    def _render_image_canvas(self):
        if self.cv_current is not None:
            if len(self.cv_current.shape) == 2:
                img_rgb = cv2.cvtColor(self.cv_current, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(img_rgb)
            orig_w, orig_h = pil_img.size

            pil_img.thumbnail((900, 600))
            new_w, new_h = pil_img.size
            self.display_ratio = orig_w / new_w

            self.tk_display = ImageTk.PhotoImage(pil_img)
            self.canvas1.delete("all")
            self.canvas1.create_image(450, 300, anchor=tk.CENTER, image=self.tk_display)
            self.img_offset = (450 - new_w // 2, 300 - new_h // 2)

    def _render_histogram(self):
        if self.cv_current is not None:
            self.ax.clear()
            self.ax.set_xlim([0, 255])
            self.ax.get_yaxis().set_visible(False)
            self.ax.tick_params(labelsize=8)
            self.ax.set_title("R G B Histogram", fontsize=9)

            if len(self.cv_current.shape) == 2:
                hist = cv2.calcHist([self.cv_current], [0], None, [256], [0, 256])
                self.ax.plot(hist, color='gray', linewidth=1.5)
                self.ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.2)
            else:
                colors = (('b', 'blue'), ('g', 'green'), ('r', 'red'))
                for i, (col_code, col_name) in enumerate(colors):
                    hist = cv2.calcHist([self.cv_current], [i], None, [256], [0, 256])
                    self.ax.plot(hist, color=col_name, linewidth=1.2, alpha=0.8)

            self.fig.tight_layout()
            self.hist_canvas.draw()

    def get_pixel_data(self, event):
        if self.cv_current is None: return

        ix = int((event.x - self.img_offset[0]) * self.display_ratio)
        iy = int((event.y - self.img_offset[1]) * self.display_ratio)

        h, w = self.cv_current.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            pixel = self.cv_current[iy, ix]
            if len(self.cv_current.shape) == 2:
                r = g = b = pixel
            else:
                b, g, r = pixel

            self.pixel_info_label.config(text=f"X: {ix}, Y: {iy} | R: {r}, G: {g}, B: {b}")

    def reset_image(self):
        if self.cv_original is not None:
            self.cv_current = self.cv_original.copy()
            self._update_interface()

    # --- Image Manipulation Filters ---
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
            res = self.cv_current.astype(np.uint16) * 2
            self.cv_current = np.clip(res, 0, 255).astype(np.uint8)
            self._update_interface()

    def apply_gray(self):
        if self.cv_current is not None and len(self.cv_current.shape) == 3:
            self.cv_current = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY)
            self._update_interface()

    def apply_star_enhance(self):
        """Enhance star structures using Thresholded Center of Mass"""
        if self.cv_current is None: return

        # Establish working grayscale image channel
        is_color = (len(self.cv_current.shape) == 3)
        working_gray = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY) if is_color else self.cv_current.copy()

        # Step 1: Establish high-pass thresholding isolation mask
        _, thresholded = cv2.threshold(working_gray, 200, 255, cv2.THRESH_BINARY)

        # Step 2: Extract distinct structural connected component regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded)

        # Create an operational mask buffer to scale star cores
        boost_mask = np.zeros_like(working_gray, dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 1 <= area <= 150:  # Filters pixel dimensions matching normal point star sizes
                component_mask = (labels == i)

                # Step 3: Compute localized moments to evaluate exact spatial Center of Mass
                moments = cv2.moments(component_mask.astype(np.uint8))
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    # Target center coordinates with a localized Gaussian glow mask expansion
                    cv2.circle(boost_mask, (cx, cy), 2, 255, -1)

        # Smooth expansion mapping out core enhancement points
        boost_mask = cv2.GaussianBlur(boost_mask, (3, 3), 0)

        # Step 4: Merge calculated highlight enhancements back into workspace layers
        if is_color:
            for ch in range(3):
                enhanced_channel = self.cv_current[:, :, ch].astype(np.uint16) + (boost_mask.astype(np.uint16) // 2)
                self.cv_current[:, :, ch] = np.clip(enhanced_channel, 0, 255).astype(np.uint8)
        else:
            enhanced_gray = self.cv_current.astype(np.uint16) + (boost_mask.astype(np.uint16) // 2)
            self.cv_current = np.clip(enhanced_gray, 0, 255).astype(np.uint8)

        self._update_interface()

    def save_image(self):
        if self.cv_current is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"), ("JPG Files", "*.jpg")])
            if save_path:
                ext = os.path.splitext(save_path)[1]
                _, buf = cv2.imencode(ext, self.cv_current)
                buf.tofile(save_path)
                messagebox.showinfo("Success", "Image saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = StarAnalyzerApp(root)
    root.mainloop()
