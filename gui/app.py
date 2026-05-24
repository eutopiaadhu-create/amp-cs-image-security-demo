"""
Main application window for the AMP-CS multi-level image encryption demo.
"""
import os
import time
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps

from core.encryptor import Encryptor
from core.decryptor import Decryptor
from ai_modules.face_detector import FaceDetector
from ai_modules.privacy_protector import PrivacyProtector


class App(ctk.CTk):
    COLORS = {
        "bg": "#0b1118",
        "panel": "#111a24",
        "panel_2": "#162231",
        "card": "#1b2735",
        "card_soft": "#202d3d",
        "stroke": "#2c3d52",
        "text": "#e7eef8",
        "muted": "#8ea0b5",
        "blue": "#2f81f7",
        "cyan": "#21c7a8",
        "amber": "#f6b44b",
        "red": "#ef5b5b",
        "green": "#34d399",
    }
    FONT = "Microsoft YaHei UI"
    DISPLAY_FONT = "Bahnschrift SemiBold"

    def __init__(self, user_role):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.user_role = user_role
        self.title(f"AI智能感知多级图像加密系统 - [{self._role_display()}]")
        self.geometry("1500x880")
        self.minsize(1360, 780)
        self.configure(fg_color=self.COLORS["bg"])

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
        self.ai_model_path = None
        self.privacy_hits = 0

        self._build_ui()
        self._load_ai_model()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _role_display(self):
        mapping = {"USER_A": "A用户", "USER_B": "B用户", "ADMIN": "管理员", "GUEST": "访客"}
        return mapping.get(self.user_role, self.user_role)

    def _load_ai_model(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_candidates = [
            "face_detect.pt",
            "best.pt",
            "model (3).pt",
            "model.pt",
            "yolov8n.pt",
        ]
        for name in model_candidates:
            for base in (os.path.join(project_root, "models"), project_root):
                path = os.path.join(base, name)
                if os.path.exists(path):
                    self.ai_model_path = path
                    self._set_ai_status("YOLO: 加载中...", self.COLORS["amber"])
                    threading.Thread(target=self._ai_load_thread, args=(path,), daemon=True).start()
                    return
        self.ai_status = "未找到模型"
        self._set_ai_status("YOLO: 未找到模型", self.COLORS["red"])

    def _ai_load_thread(self, path):
        try:
            loaded = self.face_detector.load(path)
            if loaded:
                self.ai_status = "已加载"
                self.after(0, lambda: self._set_ai_status(f"YOLO: 已加载 {os.path.basename(path)}", self.COLORS["green"]))
            else:
                self.ai_status = "加载失败"
                self.after(0, lambda: self._set_ai_status("YOLO: 加载失败", self.COLORS["red"]))
        except Exception:
            self.ai_status = "错误"
            self.after(0, lambda: self._set_ai_status("YOLO: 加载错误", self.COLORS["red"]))

    def _set_ai_status(self, text, color):
        if hasattr(self, "lbl_ai_status"):
            self.lbl_ai_status.configure(text=text, text_color=color)

    def _font(self, size, weight="normal", family=None):
        return ctk.CTkFont(family=family or self.FONT, size=size, weight=weight)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=330, fg_color=self.COLORS["panel"], corner_radius=24)
        sidebar.grid(row=0, column=0, padx=18, pady=18, sticky="nsew")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="AMP-CS",
            font=self._font(30, "bold", self.DISPLAY_FONT),
            text_color=self.COLORS["text"],
        ).pack(anchor="w", padx=24, pady=(24, 0))
        ctk.CTkLabel(
            sidebar,
            text="AI智能感知多级图像加密系统",
            font=self._font(14),
            text_color=self.COLORS["muted"],
        ).pack(anchor="w", padx=24, pady=(2, 12))
        ctk.CTkLabel(
            sidebar,
            text=f"当前角色  {self._role_display()}",
            fg_color=self.COLORS["card_soft"],
            text_color=self.COLORS["cyan"],
            corner_radius=14,
            font=self._font(13, "bold"),
            padx=12,
            pady=8,
        ).pack(anchor="w", padx=24, pady=(0, 18))

        self._build_status_panel(sidebar)
        self._build_file_panel(sidebar)
        self._build_run_panel(sidebar)

        workspace = ctk.CTkFrame(self, fg_color=self.COLORS["panel"], corner_radius=24)
        workspace.grid(row=0, column=1, padx=(0, 18), pady=18, sticky="nsew")
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(workspace, fg_color="transparent")
        header.grid(row=0, column=0, padx=22, pady=(20, 4), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="分级访问结果总览",
            font=self._font(22, "bold"),
            text_color=self.COLORS["text"],
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="B用户完整恢复源/目标，A用户展示YOLO隐私保护，访客仅可见载体隐写图",
            font=self._font(13),
            text_color=self.COLORS["muted"],
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.payload_label = ctk.CTkLabel(
            header,
            text="Payload: --",
            font=self._font(13, "bold"),
            fg_color=self.COLORS["card_soft"],
            text_color=self.COLORS["amber"],
            corner_radius=12,
            padx=12,
            pady=7,
        )
        self.payload_label.grid(row=0, column=1, rowspan=2, sticky="e")

        self.tab_view = ctk.CTkTabview(
            workspace,
            anchor="w",
            fg_color=self.COLORS["panel_2"],
            segmented_button_selected_color=self.COLORS["blue"],
            segmented_button_unselected_color=self.COLORS["card"],
            segmented_button_selected_hover_color="#3b92ff",
            segmented_button_unselected_hover_color=self.COLORS["card_soft"],
        )
        self.tab_view.grid(row=1, column=0, padx=18, pady=16, sticky="nsew")

        if self.user_role in ["ADMIN", "USER_B"]:
            self.tab_b = self.tab_view.add("B用户 · 完整访问")
        if self.user_role in ["ADMIN", "USER_B", "USER_A"]:
            self.tab_a = self.tab_view.add("A用户 · 隐私保护")
        self.tab_guest = self.tab_view.add("访客 · 载体视图")
        self._setup_tabs()

    def _build_status_panel(self, parent):
        panel = self._section(parent, "系统状态")
        self.engine_label = self._status_line(panel, "重建引擎", "AMP-Net / PyTorch", self.COLORS["green"])
        self.lbl_ai_status = self._status_line(panel, "AI感知", "YOLO: 等待加载", self.COLORS["amber"])
        self.ai_scope_label = ctk.CTkLabel(
            panel,
            text="YOLO仅作用于 A用户隐私视图；B用户完整访问不打码。",
            wraplength=260,
            justify="left",
            font=self._font(12),
            text_color=self.COLORS["muted"],
        )
        self.ai_scope_label.pack(anchor="w", padx=16, pady=(6, 14))

    def _build_file_panel(self, parent):
        panel = self._section(parent, "输入素材")
        self._file_picker(panel, "源图像", "source", 3)
        self._file_picker(panel, "目标图像", "target", 3)
        self._file_picker(panel, "载体图像", "carrier", 1)

    def _build_run_panel(self, parent):
        panel = self._section(parent, "运行控制")
        self.privacy_var = ctk.StringVar(value="on")
        ctk.CTkSwitch(
            panel,
            text="A用户视图启用 YOLO 隐私保护",
            variable=self.privacy_var,
            onvalue="on",
            offvalue="off",
            progress_color=self.COLORS["cyan"],
            button_color=self.COLORS["text"],
            font=self._font(12),
        ).pack(anchor="w", padx=16, pady=(8, 12))

        self.run_btn = ctk.CTkButton(
            panel,
            text="开始加密与重建",
            height=44,
            fg_color=self.COLORS["blue"],
            hover_color="#3b92ff",
            corner_radius=14,
            font=self._font(15, "bold"),
            command=self.start_process,
        )
        self.run_btn.pack(padx=16, pady=(0, 12), fill="x")

        self.progress = ctk.CTkProgressBar(panel, progress_color=self.COLORS["cyan"])
        self.progress.set(0)
        self.progress.pack(padx=16, pady=(0, 10), fill="x")

        self.status_label = ctk.CTkLabel(panel, text="状态: 就绪", font=self._font(12), text_color=self.COLORS["muted"])
        self.status_label.pack(anchor="w", padx=16)
        self.time_label = ctk.CTkLabel(panel, text="耗时: --", font=self._font(12), text_color=self.COLORS["muted"])
        self.time_label.pack(anchor="w", padx=16, pady=(2, 12))

        self.present_btn = ctk.CTkButton(
            panel,
            text="打开过程演示",
            fg_color=self.COLORS["card_soft"],
            hover_color=self.COLORS["stroke"],
            corner_radius=14,
            command=self.open_presentation,
            state="disabled",
        )
        self.present_btn.pack(padx=16, pady=(0, 16), fill="x")

    def _section(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=self.COLORS["card"], corner_radius=18)
        frame.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkLabel(frame, text=title, font=self._font(15, "bold"), text_color=self.COLORS["text"]).pack(
            anchor="w", padx=16, pady=(14, 8)
        )
        return frame

    def _status_line(self, parent, label, value, color):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(row, text=label, font=self._font(12), text_color=self.COLORS["muted"]).pack(side="left")
        value_lbl = ctk.CTkLabel(row, text=value, font=self._font(12, "bold"), text_color=color)
        value_lbl.pack(side="right")
        return value_lbl

    def _file_picker(self, parent, label, ftype, count):
        box = ctk.CTkFrame(parent, fg_color=self.COLORS["card_soft"], corner_radius=14)
        box.pack(fill="x", padx=16, pady=(0, 10))
        top = ctk.CTkFrame(box, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(top, text=label, font=self._font(12, "bold"), text_color=self.COLORS["text"]).pack(side="left")
        ctk.CTkButton(
            top,
            text=f"选择{count}张" if count > 1 else "选择",
            width=72,
            height=26,
            fg_color=self.COLORS["blue"],
            hover_color="#3b92ff",
            corner_radius=10,
            font=self._font(11),
            command=lambda: self.select_files(count, ftype),
        ).pack(side="right")
        value = ctk.CTkLabel(
            box,
            text="未选择",
            wraplength=250,
            justify="left",
            anchor="w",
            font=self._font(11),
            text_color=self.COLORS["muted"],
        )
        value.pack(fill="x", padx=12, pady=(0, 10))
        if ftype == "source":
            self.lbl_source = value
        elif ftype == "target":
            self.lbl_target = value
        else:
            self.lbl_carrier = value

    def _setup_tabs(self):
        if hasattr(self, "tab_b"):
            self._setup_b_tab()
        if hasattr(self, "tab_a"):
            self._setup_a_tab()
        self._setup_guest_tab()

    def _setup_b_tab(self):
        tab = self.tab_b
        for col in range(3):
            tab.grid_columnconfigure(col, weight=1, uniform="b_images")
        tab.grid_columnconfigure(3, weight=0)
        tab.grid_rowconfigure((0, 1), weight=1, uniform="b_rows")

        self.b_images = []
        self.b_footers = []
        for i in range(3):
            card = self._image_card(tab, f"源图像 {i + 1}", "B用户 · 直接源measurement重建", self.COLORS["amber"])
            card["frame"].grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            self.b_images.append(card["image"])
            self.b_footers.append(card["footer"])

        for i in range(3):
            card = self._image_card(tab, f"目标图像 {i + 1}", "A/B用户 · 目标measurement重建", self.COLORS["cyan"])
            card["frame"].grid(row=1, column=i, padx=10, pady=10, sticky="nsew")
            self.b_images.append(card["image"])
            self.b_footers.append(card["footer"])

        info = ctk.CTkFrame(tab, width=250, fg_color=self.COLORS["card"], corner_radius=20)
        info.grid(row=0, column=3, rowspan=2, padx=(10, 12), pady=10, sticky="nsew")
        info.grid_propagate(False)
        ctk.CTkLabel(info, text="质量指标", font=self._font(18, "bold"), text_color=self.COLORS["text"]).pack(
            anchor="w", padx=18, pady=(18, 4)
        )
        ctk.CTkLabel(
            info,
            text="源图已采用方案1：B用户持有源图真实压缩测量，PSNR不再依赖P矩阵近似。",
            wraplength=205,
            justify="left",
            font=self._font(12),
            text_color=self.COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(0, 14))
        self.metrics_lbl = ctk.CTkLabel(
            info,
            text="源图平均\nPSNR: --\nSSIM: --\n\n目标平均\nPSNR: --\nSSIM: --",
            font=self._font(14, "bold"),
            justify="left",
            text_color=self.COLORS["text"],
        )
        self.metrics_lbl.pack(anchor="w", padx=18, pady=(0, 16))
        self.scheme_lbl = ctk.CTkLabel(
            info,
            text="重建链路: 等待运行",
            wraplength=205,
            justify="left",
            font=self._font(12),
            text_color=self.COLORS["amber"],
        )
        self.scheme_lbl.pack(anchor="w", padx=18, pady=(0, 16))

    def _setup_a_tab(self):
        tab = self.tab_a
        for col in range(3):
            tab.grid_columnconfigure(col, weight=1, uniform="a_images")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)

        self.a_images = []
        self.a_footers = []
        for i in range(3):
            card = self._image_card(tab, f"A用户可见目标 {i + 1}", "YOLO检测后隐私保护", self.COLORS["blue"])
            card["frame"].grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            self.a_images.append(card["image"])
            self.a_footers.append(card["footer"])

        status = ctk.CTkFrame(tab, fg_color=self.COLORS["card"], corner_radius=18)
        status.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="ew")
        status.grid_columnconfigure(0, weight=1)
        self.ai_privacy_status = ctk.CTkLabel(
            status,
            text="AI隐私保护: 等待运行",
            font=self._font(13, "bold"),
            text_color=self.COLORS["amber"],
        )
        self.ai_privacy_status.grid(row=0, column=0, sticky="w", padx=16, pady=12)
        ctk.CTkLabel(
            status,
            text="说明: 此页模拟低权限用户视图；即使当前登录B用户，也会展示A用户应看到的打码结果。",
            font=self._font(12),
            text_color=self.COLORS["muted"],
        ).grid(row=0, column=1, sticky="e", padx=16, pady=12)

    def _setup_guest_tab(self):
        tab = self.tab_guest
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=0)
        tab.grid_rowconfigure(0, weight=1)
        card = self._image_card(tab, "访客可见载体", "DCT域隐写后的载体图像", self.COLORS["green"], image_size=(640, 520))
        card["frame"].grid(row=0, column=0, padx=14, pady=14, sticky="nsew")
        self.guest_image = card["image"]
        self.guest_footer = card["footer"]

        info = ctk.CTkFrame(tab, width=300, fg_color=self.COLORS["card"], corner_radius=20)
        info.grid(row=0, column=1, padx=(4, 14), pady=14, sticky="nsew")
        info.grid_propagate(False)
        ctk.CTkLabel(info, text="访客权限", font=self._font(18, "bold"), text_color=self.COLORS["text"]).pack(
            anchor="w", padx=18, pady=(18, 4)
        )
        ctk.CTkLabel(
            info,
            text="访客只看到视觉上接近原图的载体图像；没有R矩阵、置乱索引和AMP测量数据，无法重建目标或源图。",
            wraplength=245,
            justify="left",
            font=self._font(12),
            text_color=self.COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(0, 12))

    def _image_card(self, parent, title, subtitle, accent, image_size=(270, 230)):
        frame = ctk.CTkFrame(parent, fg_color=self.COLORS["card"], corner_radius=20, border_width=1, border_color=self.COLORS["stroke"])
        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(12, 6))
        ctk.CTkLabel(top, text=title, font=self._font(14, "bold"), text_color=self.COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(top, text=subtitle, font=self._font(11), text_color=accent).pack(anchor="w", pady=(1, 0))
        image = ctk.CTkLabel(frame, text="等待运行", text_color=self.COLORS["muted"], fg_color="#0f1722", corner_radius=16)
        image.pack(expand=True, fill="both", padx=14, pady=(4, 8))
        footer = ctk.CTkLabel(frame, text="--", font=self._font(11), text_color=self.COLORS["muted"])
        footer.pack(anchor="w", padx=14, pady=(0, 12))
        image.preview_size = image_size
        return {"frame": frame, "image": image, "footer": footer}

    def select_files(self, num, ftype):
        paths = filedialog.askopenfilenames(title=f"选择 {num} 张图像", filetypes=[("图像", "*.png *.jpg *.bmp *.jpeg")])
        if len(paths) != num:
            if paths:
                messagebox.showwarning("选择错误", f"请选择 {num} 个文件。")
            return
        names = self._format_selected_names(paths)
        if ftype == "source":
            self.source_paths = list(paths)
            self.lbl_source.configure(text=names, text_color=self.COLORS["text"])
        elif ftype == "target":
            self.target_paths = list(paths)
            self.lbl_target.configure(text=names, text_color=self.COLORS["text"])
        elif ftype == "carrier":
            self.carrier_path = paths[0]
            self.lbl_carrier.configure(text=names, text_color=self.COLORS["text"])

    def _format_selected_names(self, paths):
        names = [os.path.basename(p) for p in paths]
        return "\n".join(names[:3])

    def start_process(self):
        if not (self.source_paths and self.target_paths and self.carrier_path):
            messagebox.showerror("输入错误", "请选择所有必需的图像。")
            return
        self.run_btn.configure(state="disabled", text="处理中...")
        self.present_btn.configure(state="disabled")
        self.status_label.configure(text="状态: 生成目标/源图测量并嵌入...")
        self.progress.start()
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            t0 = time.time()
            self.encryption_result = self.encryptor.encrypt(self.source_paths, self.target_paths, self.carrier_path, self.output_dir)
            self.after(0, lambda: self.status_label.configure(text="状态: AMP解密重建中..."))
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
        self.run_btn.configure(state="normal", text="开始加密与重建")
        self.present_btn.configure(state="normal")
        self.status_label.configure(text="状态: 完成")
        self.time_label.configure(text=f"耗时: {elapsed:.1f}s")
        self._load_results()

    def _on_error(self, msg):
        self.progress.stop()
        self.progress.set(0)
        self.run_btn.configure(state="normal", text="开始加密与重建")
        self.status_label.configure(text="状态: 错误")
        messagebox.showerror("处理错误", msg)

    def _load_results(self):
        self.privacy_hits = 0
        if hasattr(self, "tab_b"):
            source_metrics = []
            target_metrics = []
            for i in range(3):
                source_path = os.path.join(self.output_dir, f"recovered_B_source_{i + 1}.png")
                target_path = os.path.join(self.output_dir, f"recovered_B_target_{i + 1}.png")
                self._show_img(self.b_images[i], source_path, self.b_images[i].preview_size, footer=self.b_footers[i], footer_text="方案1 · 源图真实measurement")
                self._show_img(
                    self.b_images[i + 3],
                    target_path,
                    self.b_images[i + 3].preview_size,
                    footer=self.b_footers[i + 3],
                    footer_text="AMP目标measurement",
                )
                source_metrics.append(self._metrics_for(self.source_paths[i], source_path))
                target_metrics.append(self._metrics_for(self.target_paths[i], target_path))
            self._update_metrics(source_metrics, target_metrics)

        if hasattr(self, "tab_a"):
            for i in range(3):
                path = os.path.join(self.output_dir, f"recovered_A_{i + 1}.png")
                detect_path = self.target_paths[i] if i < len(self.target_paths) else None
                self._show_img(
                    self.a_images[i],
                    path,
                    self.a_images[i].preview_size,
                    apply_privacy=True,
                    detect_path=detect_path,
                    footer=self.a_footers[i],
                )
            if self.privacy_var.get() != "on":
                msg = "AI隐私保护: 已关闭"
                color = self.COLORS["muted"]
            elif not self.face_detector.is_loaded:
                msg = "AI隐私保护: YOLO未加载，未执行检测"
                color = self.COLORS["red"]
            else:
                msg = f"AI隐私保护: 已处理 {self.privacy_hits} 个敏感区域"
                color = self.COLORS["green"] if self.privacy_hits else self.COLORS["amber"]
            self.ai_privacy_status.configure(text=msg, text_color=color)

        stego_path = os.path.join(self.output_dir, "11_carrier_steganographic.png")
        self._show_img(self.guest_image, stego_path, self.guest_image.preview_size, footer=self.guest_footer, footer_text="访客仅可见载体")

        if self.encryption_result:
            m = self.encryption_result.get("M", "--")
            w = self.encryption_result.get("measurement_width", "--")
            ew = self.encryption_result.get("embed_width", "--")
            self.payload_label.configure(text=f"Payload: {m}x{ew}  (目标/源 {w}+{w})")
            if hasattr(self, "scheme_lbl"):
                self.scheme_lbl.configure(text=f"重建链路: 目标measurement + 源measurement直接重建\n嵌入区域: {m} x {ew}")

    def _metrics_for(self, original_path, recovered_path):
        if not original_path or not os.path.exists(recovered_path):
            return None
        original = self.encryptor.load_image_gray(original_path)
        recovered = np.array(Image.open(recovered_path).convert("L"), dtype=np.float64)
        return self.decryptor.compute_metrics(original, recovered)

    def _update_metrics(self, source_metrics, target_metrics):
        def avg(metrics, key):
            vals = [m[key] for m in metrics if m is not None]
            return float(np.mean(vals)) if vals else float("nan")

        src_psnr = avg(source_metrics, "psnr")
        src_ssim = avg(source_metrics, "ssim")
        tgt_psnr = avg(target_metrics, "psnr")
        tgt_ssim = avg(target_metrics, "ssim")
        self.metrics_lbl.configure(
            text=(
                f"源图平均\nPSNR: {src_psnr:.2f} dB\nSSIM: {src_ssim:.4f}\n\n"
                f"目标平均\nPSNR: {tgt_psnr:.2f} dB\nSSIM: {tgt_ssim:.4f}"
            )
        )

    def _show_img(self, label, path, size, apply_privacy=False, detect_path=None, footer=None, footer_text=None):
        if not os.path.exists(path):
            label.configure(image=None, text="图片未找到")
            if footer:
                footer.configure(text="文件不存在", text_color=self.COLORS["red"])
            return
        try:
            img = Image.open(path).convert("RGB")
            protection_text = footer_text
            if apply_privacy:
                img, hit_count, protection_text = self._apply_privacy(img, path, detect_path)
                self.privacy_hits += hit_count
            preview = self._prepare_preview(img, size)
            ctk_img = ctk.CTkImage(light_image=preview, dark_image=preview, size=size)
            label.configure(image=ctk_img, text="")
            label.image = ctk_img
            if footer:
                footer.configure(text=protection_text or footer_text or "--", text_color=self.COLORS["muted"])
        except Exception:
            label.configure(image=None, text="显示错误")
            if footer:
                footer.configure(text="显示错误", text_color=self.COLORS["red"])

    def _apply_privacy(self, img, reconstructed_path, detect_path):
        if self.privacy_var.get() != "on":
            return img, 0, "AI保护已关闭"
        if not self.face_detector.is_loaded:
            return img, 0, "YOLO未加载"

        detections = []
        if detect_path and os.path.exists(detect_path):
            detections = self.face_detector.detect_with_scaling(detect_path, (img.height, img.width))
        if not detections:
            detections = self.face_detector.detect(image_path=reconstructed_path)

        if not detections:
            return img, 0, "YOLO未检出敏感区域"

        protector = PrivacyProtector(method="mosaic", mosaic_size=10)
        protected = protector.protect_pil(img, detections)
        return protected, len(detections), f"YOLO已保护 {len(detections)} 个区域"

    def _prepare_preview(self, img, size):
        canvas = Image.new("RGB", size, "#0f1722")
        preview = ImageOps.contain(img.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        x = (size[0] - preview.width) // 2
        y = (size[1] - preview.height) // 2
        canvas.paste(preview, (x, y))
        return canvas

    def open_presentation(self):
        from gui.presentation import PresentationWindow

        PresentationWindow(self, self.output_dir)

    def on_closing(self):
        if messagebox.askokcancel("退出", "确定退出？"):
            self.destroy()
