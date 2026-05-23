"""
2D-CS AI-Driven Intelligent Perception Multi-level Image Encryption System
Main Entry Point - Pure Python (No MATLAB dependency)

Usage:
    python main_app.py

Login credentials:
    - A用户: username='a', password='123'
    - B用户: username='b', password='456'
    - 管理员: username='admin', password='root'
    - 访客: click "访客登录"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk
from tkinter import messagebox

from core.permission import PermissionManager

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class LoginWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("系统登录")
        self.geometry("380x320")
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.permission_mgr = PermissionManager()
        self.user_role = None

        frame = ctk.CTkFrame(self)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(
            frame, text="AI智能感知加密系统",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 5))
        ctk.CTkLabel(
            frame, text="Multi-level Image Encryption",
            font=ctk.CTkFont(size=12), text_color="gray"
        ).pack(pady=(0, 15))

        self.user_entry = ctk.CTkEntry(frame, width=220, placeholder_text="用户名 (A, B 或 admin)")
        self.user_entry.pack(pady=8)
        self.pass_entry = ctk.CTkEntry(frame, width=220, placeholder_text="密码", show="*")
        self.pass_entry.pack(pady=8)

        ctk.CTkButton(frame, text="登录", width=220, command=self.login).pack(pady=12)
        ctk.CTkButton(frame, text="访客登录", width=220, fg_color="gray", command=self.guest_login).pack(pady=5)

    def login(self):
        user = self.permission_mgr.authenticate(self.user_entry.get(), self.pass_entry.get())
        if user:
            self.user_role = user.level.name
            self.destroy()
        else:
            messagebox.showerror("登录失败", "用户名或密码错误。")

    def guest_login(self):
        self.permission_mgr.guest_login()
        self.user_role = "GUEST"
        self.destroy()

    def _on_close(self):
        self.user_role = "exit"
        self.destroy()

    def get_role(self):
        self.master.wait_window(self)
        return self.user_role


def main():
    root = ctk.CTk()
    root.withdraw()

    login = LoginWindow(root)
    role = login.get_role()

    if role and role != "exit":
        root.destroy()
        from gui.app import App
        app = App(role)
        app.mainloop()
    else:
        root.destroy()


if __name__ == "__main__":
    main()
