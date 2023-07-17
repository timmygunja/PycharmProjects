from tkinter import *
from tkinter import messagebox
import hashlib
import hmac

root = Tk()
root.geometry("600x400")
root.title("Урусов Тимур ПИ19-2")


def clear():
	my_text.delete(1.0, END)
	my_entry.delete(1.0, END)


def encrypt():
	key = my_text.get(1.0, END)
	message = my_entry.get(1.0, END)

	# Генерация хэша
	signature = hmac.new(key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()

	# Сообщение
	messagebox.showwarning("Успех!", "Ваш HMAC-пароль: " + signature)


my_frame = Frame(root)
border_color = Frame()
my_frame.pack(pady=20)

enc_button = Button(my_frame, text="Зашифровать", font=("Helvetica", 18), command=encrypt)
enc_button.grid(row=0, column=0)

clear_button = Button(my_frame, text="Очистить", font=("Helvetica", 18), command=clear)
clear_button.grid(row=0, column=2, padx=10)

enc_label = Label(root, text="Введите ключ для HMAC-шифрования (стоит значение по умолчанию)", font=("Helvetica", 14))
enc_label.pack()

my_text = Text(root, width=57, height=10)
my_text.pack(pady=10)
my_text.insert(END, "8oe0i89o7es243t5s234")


password_label = Label(root, text="Введите сообщение для шифровки (стоит значение по умолчанию)", font=("Helvetica", 14))
password_label.pack()

my_entry = Text(root, font=("Helvetica", 14), width=49, height=10)
my_entry.pack(pady=10)
my_entry.insert(END, "Текст, который нужно зашифровать")


root.mainloop()
