from tkinter import *
import tkinter
from PIL import Image,ImageTk
import subprocess

root = tkinter.Tk()
root.title("Face mask detection and alert system")
root.configure(bg='#87cefa')

canvas=tkinter.Canvas(root,width=600,height=400,)
canvas.grid(columnspan=3,rowspan=3)

logo = Image.open('pp.png')
logo = ImageTk.PhotoImage(logo,width=300,height=300)
logo_label = tkinter.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=1, row=0)

instruction = tkinter.Label(root, text=" 'we ensure a safe environment for you' ",
                            font="Raleway", anchor = "center", justify = "center",
                            padx=10,pady=10,bg="#87cefa")
instruction.grid(column=1, row=10,padx=10,pady=10)

def Click():

    subprocess.call([r"C:\Users\admin\PycharmProjects\webcam test\fmdas.exe"])


browse_btn = tkinter.Button(root,text = "Start", font="Calibri", bg="#64e764",
                       fg="white", height=1, width=15, anchor = "center",
                       bd=2, justify = "center",command= Click,
                            padx=10,pady=10)

browse_btn.grid(column=1, row=20,padx=10,pady=10)


root.mainloop()
