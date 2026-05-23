"""Presentation window for step-by-step encryption process demonstration."""
import os
import customtkinter as ctk
from PIL import Image


class PresentationWindow(ctk.CTkToplevel):
    def __init__(self, master, image_folder):
        super().__init__(master)
        self.title("加密过程演示")
        self.geometry("1000x700")
        self.transient(master)
        self.image_folder = image_folder
        self.current_step = -1
        self.after_id = None

        self.steps = [
            [["01_source_image_1.png"], ["07_encrypted_data_1.png"],
             "Step 1: 压缩感知加密",
             "原始图像经过混沌压缩感知后变为不可识别的噪声数据"],
            [["10_carrier_original.png"], ["11_carrier_steganographic.png"],
             "Step 2: DCT域隐写嵌入",
             "加密数据嵌入载体图像，视觉上几乎无差异"],
            [["91_dct_before_embedding.png"], ["92_dct_after_embedding.png"],
             "Step 3: DCT系数变化",
             "频域中可观察到嵌入信息的痕迹"],
            [["92_dct_after_embedding.png"], ["93_dct_difference_secret.png"],
             "Step 4: 秘密信息分离",
             "DCT系数差值揭示了隐藏的加密信息"],
            [["11_carrier_steganographic.png"], ["recovered_A_1.png"],
             "Step 5: A用户解密",
             "部分权限用户恢复目标图像"],
            [["recovered_B_target_1.png"], ["recovered_B_source_1.png"],
             "Step 6: B用户解密",
             "完全权限用户恢复源图像"],
        ]

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.title_lbl = ctk.CTkLabel(self, text="点击开始演示",
                                      font=("Arial", 22, "bold"))
        self.title_lbl.grid(row=0, column=0, columnspan=2, pady=15)

        self.left_lbl = ctk.CTkLabel(self, text="")
        self.left_lbl.grid(row=1, column=0, sticky="nsew", padx=10)
        self.right_lbl = ctk.CTkLabel(self, text="")
        self.right_lbl.grid(row=1, column=1, sticky="nsew", padx=10)

        self.desc_lbl = ctk.CTkLabel(self, text="", font=("Arial", 14))
        self.desc_lbl.grid(row=2, column=0, columnspan=2, pady=10)

        self.btn = ctk.CTkButton(self, text="开始演示", command=self.start)
        self.btn.grid(row=3, column=0, columnspan=2, pady=15)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def start(self):
        self.btn.configure(state="disabled", text="演示中...")
        self.current_step = -1
        self._next()

    def _next(self):
        self.current_step += 1
        if self.current_step >= len(self.steps):
            self.title_lbl.configure(text="演示结束")
            self.desc_lbl.configure(text="")
            self.btn.configure(state="normal", text="重新演示")
            return
        left_f, right_f, title, desc = self.steps[self.current_step]
        self.title_lbl.configure(text=title)
        self.desc_lbl.configure(text=desc)
        self._display(self.left_lbl, left_f[0])
        self._display(self.right_lbl, right_f[0])
        self.after_id = self.after(4000, self._next)

    def _display(self, lbl, filename):
        path = os.path.join(self.image_folder, filename)
        if os.path.exists(path):
            img = Image.open(path)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(380, 380))
            lbl.configure(image=ctk_img, text="")
            lbl.image = ctk_img
        else:
            lbl.configure(image=None, text=f"未找到: {filename}")

    def _close(self):
        if self.after_id:
            self.after_cancel(self.after_id)
        self.destroy()
