"""
Main Application Window for the encryption system.
"""
import os
import time
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

from core.encryptor import Encryptor
from core.decryptor import Decryptor
from ai_modules.face_detector import FaceDetector
from ai_modules.privacy_protector import PrivacyProtector


class App(ctk.CTk):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.title(f"AI智能感知多级图像加密系统 - [{self._role_display()}]")
        self.geometry("1300x750")

        self.source_paths = []
        self.target_paths = []
        self.carrier_path = ""
        self.output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "2DCS_AI_Output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.encryptor = Encryptor()
        self.decryptor = Decryptor()
        self.encryption_result = None

        self.face_detector = FaceDetector(confidence=0.5)
        self.ai_status = "未加载"
        self._load_ai_model()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _role_display(self):
        mapping = {"USER_A": "A用户", "USER_B": "B用户", "ADMIN": "管理员", "GUEST": "访客"}
        return mapping.get(self.user_role, self.user_role)

    def _load_ai_model(self):
        model_candidates = ["model (3).pt", "model.pt", "yolov8n.pt"]
        for m in model_candidates:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), m)
            if os.path.exists(path):
                threading.Thread(target=self._ai_load_thread, args=(path,), daemon=True).start()
                return
        self.ai_status = "未找到模型"

    def _ai_load_thread(self, path):
        try:
            if self.face_detector.load(path):
                self.ai_status = "已加载"
                self.after(0, lambda: self.lbl_ai_status.configure(text="AI模型: 已加载", text_color="green"))
            else:
                self.ai_status = "加载失败"
                self.after(0, lambda: self.lbl_ai_status.configure(text="AI模型: 加载失败", text_color="red"))
        except Exception:
            self.ai_status = "错误"
            self.after(0, lambda: self.lbl_ai_status.configure(text="AI模型: 错误", text_color="red"))

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, width=320, corner_radius=10)
        left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left.grid_propagate(False)

        ctk.CTkLabel(left, text="1. 系统状态", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5), padx=10)
        ctk.CTkLabel(left, text="引擎: 纯Python (无MATLAB依赖)", text_color="green").pack(padx=20)
        self.lbl_ai_status = ctk.CTkLabel(left, text="AI模型: 加载中...", text_color="yellow")
        self.lbl_ai_status.pack(padx=20, pady=(0, 10))

        ctk.CTkLabel(left, text="2. 选择图像", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5), padx=10)

        ctk.CTkButton(left, text="选择源图像 (3张)", command=lambda: self.select_files(3, 'source')).pack(pady=5, padx=20, fill="x")
        self.lbl_source = ctk.CTkLabel(left, text="未选择", wraplength=280, anchor="w")
        self.lbl_source.pack(padx=20, fill="x")

        ctk.CTkButton(left, text="选择目标图像 (3张)", command=lambda: self.select_files(3, 'target')).pack(pady=5, padx=20, fill="x")
        self.lbl_target = ctk.CTkLabel(left, text="未选择", wraplength=280, anchor="w")
        self.lbl_target.pack(padx=20, fill="x")

        ctk.CTkButton(left, text="选择载体图像 (1张)", command=lambda: self.select_files(1, 'carrier')).pack(pady=5, padx=20, fill="x")
        self.lbl_carrier = ctk.CTkLabel(left, text="未选择", wraplength=280, anchor="w")
        self.lbl_carrier.pack(padx=20, fill="x")

        ctk.CTkLabel(left, text="3. 运行", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5), padx=10)

        self.privacy_var = ctk.StringVar(value="on")
        ctk.CTkSwitch(left, text="AI隐私保护", variable=self.privacy_var, onvalue="on", offvalue="off").pack(pady=5, padx=20)

        self.run_btn = ctk.CTkButton(left, text="开始加密处理", height=40, font=ctk.CTkFont(size=15), command=self.start_process)
        self.run_btn.pack(pady=10, padx=20, fill="x")

        self.progress = ctk.CTkProgressBar(left)
        self.progress.set(0)
        self.progress.pack(pady=5, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(left, text="状态: 就绪")
        self.status_label.pack(pady=5, padx=20)
        self.time_label = ctk.CTkLabel(left, text="")
        self.time_label.pack(padx=20)

        self.present_btn = ctk.CTkButton(left, text="过程演示", command=self.open_presentation, state="disabled")
        self.present_btn.pack(pady=10, padx=20, fill="x")

        right = ctk.CTkFrame(self, corner_radius=10)
        right.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.tab_view = ctk.CTkTabview(right, anchor="w")
        self.tab_view.pack(expand=True, fill="both", padx=10, pady=10)

        if self.user_role in ["ADMIN", "USER_B"]:
            self.tab_b = self.tab_view.add("B用户 (完全访问)")
        if self.user_role in ["ADMIN", "USER_B", "USER_A"]:
            self.tab_a = self.tab_view.add("A用户 (部分访问)")
        self.tab_guest = self.tab_view.add("访客视图")
        self._setup_tabs()

    def _setup_tabs(self):
        if hasattr(self, 'tab_b'):
            self.b_images = []
            for i in range(6):
                f = ctk.CTkFrame(self.tab_b)
                f.grid(row=i // 3, column=i % 3, padx=8, pady=8, sticky="nsew")
                self.tab_b.grid_rowconfigure(i // 3, weight=1)
                self.tab_b.grid_columnconfigure(i % 3, weight=1)
                title = f"源图像 {i+1}" if i < 3 else f"目标图像 {i-2}"
                ctk.CTkLabel(f, text=title).pack(pady=(3, 0))
                lbl = ctk.CTkLabel(f, text="")
                lbl.pack(expand=True, fill="both", padx=3, pady=3)
                self.b_images.append(lbl)
            self.metrics_lbl = ctk.CTkLabel(self.tab_b, text="PSNR: --\nSSIM: --", font=ctk.CTkFont(size=14), justify="left")
            self.metrics_lbl.grid(row=0, column=3, rowspan=2, padx=15, sticky="nsew")

        if hasattr(self, 'tab_a'):
            self.a_images = []
            self.tab_a.grid_columnconfigure((0, 1, 2), weight=1)
            self.tab_a.grid_rowconfigure(0, weight=1)
            for i in range(3):
                f = ctk.CTkFrame(self.tab_a)
                f.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
                ctk.CTkLabel(f, text=f"目标图像 {i+1}").pack(pady=(3, 0))
                lbl = ctk.CTkLabel(f, text="")
                lbl.pack(expand=True, fill="both", padx=3, pady=3)
                self.a_images.append(lbl)

        gf = ctk.CTkFrame(self.tab_guest)
        gf.pack(expand=True, fill="both", padx=10, pady=10)
        ctk.CTkLabel(gf, text="隐写图像 (载体)").pack(pady=(5, 0))
        self.guest_image = ctk.CTkLabel(gf, text="")
        self.guest_image.pack(expand=True, fill="both", padx=5, pady=5)

    def select_files(self, num, ftype):
        paths = filedialog.askopenfilenames(title=f"选择 {num} 张图像", filetypes=[("图像", "*.png *.jpg *.bmp *.jpeg")])
        if len(paths) != num:
            if paths:
                messagebox.showwarning("选择错误", f"请选择 {num} 个文件。")
            return
        names = "\n".join([os.path.basename(p) for p in paths])
        if ftype == 'source':
            self.source_paths = list(paths)
            self.lbl_source.configure(text=names)
        elif ftype == 'target':
            self.target_paths = list(paths)
            self.lbl_target.configure(text=names)
        elif ftype == 'carrier':
            self.carrier_path = paths[0]
            self.lbl_carrier.configure(text=names)

    def start_process(self):
        if not (self.source_paths and self.target_paths and self.carrier_path):
            messagebox.showerror("输入错误", "请选择所有必需的图像。")
            return
        self.run_btn.configure(state="disabled", text="处理中...")
        self.present_btn.configure(state="disabled")
        self.status_label.configure(text="状态: 加密处理中...")
        self.progress.start()
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            t0 = time.time()
            self.encryption_result = self.encryptor.encrypt(self.source_paths, self.target_paths, self.carrier_path, self.output_dir)
            self.after(0, lambda: self.status_label.configure(text="状态: 解密重建中..."))
            self.decryptor.decrypt_user_a(self.encryption_result, self.output_dir)
            self.decryptor.decrypt_user_b(self.encryption_result, self.output_dir)
            elapsed = time.time() - t0
            self.after(0, self._on_done, elapsed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, self._on_error, str(e))

    def _on_done(self, elapsed):
        self.progress.stop()
        self.progress.set(1)
        self.run_btn.configure(state="normal", text="开始加密处理")
        self.present_btn.configure(state="normal")
        self.status_label.configure(text="状态: 完成!")
        self.time_label.configure(text=f"耗时: {elapsed:.1f}s")
        self._load_results()

    def _on_error(self, msg):
        self.progress.stop()
        self.progress.set(0)
        self.run_btn.configure(state="normal", text="开始加密处理")
        self.status_label.configure(text="状态: 错误!")
        messagebox.showerror("处理错误", msg)

    def _load_results(self):
        if hasattr(self, 'tab_b'):
            for i in range(3):
                self._show_img(self.b_images[i], os.path.join(self.output_dir, f'recovered_B_source_{i+1}.png'), (220, 220))
                self._show_img(self.b_images[i+3], os.path.join(self.output_dir, f'recovered_B_target_{i+1}.png'), (220, 220))
            src = self.encryptor.load_image_gray(self.source_paths[0])
            rec_path = os.path.join(self.output_dir, 'recovered_B_source_1.png')
            if os.path.exists(rec_path):
                rec = np.array(Image.open(rec_path).convert('L'), dtype=np.float64)
                metrics = self.decryptor.compute_metrics(src, rec)
                self.metrics_lbl.configure(text=f"PSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.4f}")

        if hasattr(self, 'tab_a'):
            for i in range(3):
                path = os.path.join(self.output_dir, f'recovered_A_{i+1}.png')
                self._show_img(self.a_images[i], path, (220, 220), apply_privacy=True)

        stego_path = os.path.join(self.output_dir, '11_carrier_steganographic.png')
        self._show_img(self.guest_image, stego_path, (450, 450))

    def _show_img(self, label, path, size, apply_privacy=False):
        if not os.path.exists(path):
            label.configure(image=None, text="图片未找到")
            return
        try:
            img = Image.open(path)
            if apply_privacy and self.privacy_var.get() == "on" and self.face_detector.is_loaded:
                if self.user_role not in ["ADMIN", "USER_B"]:
                    protector = PrivacyProtector(method='mosaic', mosaic_size=10)
                    detections = self.face_detector.detect(image_path=path)
                    if detections:
                        img = protector.protect_pil(img, detections)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
            label.configure(image=ctk_img, text="")
            label.image = ctk_img
        except Exception:
            label.configure(image=None, text="显示错误")

    def open_presentation(self):
        from gui.presentation import PresentationWindow
        PresentationWindow(self, self.output_dir)

    def on_closing(self):
        if messagebox.askokcancel("退出", "确定退出？"):
            self.destroy()
