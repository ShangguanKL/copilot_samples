import tkinter as tk
from tkinter import ttk

class GUI:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack()
        self.combobox = ttk.Combobox(self.frame, values=["a", "b", "c"])
        self.combobox.pack()
        self.combobox.bind("<<ComboboxSelected>>", self.on_select)
        self.entries = []
        self.params = {"a": 5, "b": 3, "c": 3}
        self.values = {"a": [], "b": [], "c": []}

    def on_select(self, event):
        for entry in self.entries:
            entry.pack_forget()
        self.entries.clear()
        func = self.combobox.get()
        for i in range(self.params[func]):
            entry = tk.Entry(self.frame)
            entry.pack()
            if i < len(self.values[func]):
                entry.insert(0, self.values[func][i])
            self.entries.append(entry)

    def on_close(self):
        for func in ["a", "b", "c"]:
            self.values[func] = [entry.get() for entry in self.entries]
        self.master.destroy()

root = tk.Tk()
gui = GUI(root)
root.protocol("WM_DELETE_WINDOW", gui.on_close)
root.mainloop()
