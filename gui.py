import tkinter as tk
import subprocess

def get_output():
    # 使用 subprocess.Popen() 方法启动另一个 Python 程序
    process = subprocess.Popen(['python', 'other.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 循环读取进程的输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            # 在文本框中显示输出
            text.insert(tk.END, output.decode('utf-8'))
            text.see(tk.END)
            # 更新 GUI
            root.update_idletasks()

# 创建 Tkinter 窗口
root = tk.Tk()
root.geometry('400x300')

# 创建文本框用于显示输出
text = tk.Text(root)
text.pack(expand=True, fill=tk.BOTH)

# 创建按钮用于启动获取输出的函数
button = tk.Button(root, text='Get Output', command=get_output)
button.pack()

root.mainloop()
