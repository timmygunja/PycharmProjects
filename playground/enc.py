from tkinter import *
import pybase64
from tkinter import messagebox

root = Tk()
root.geometry("600x400")
root.title("Урусов Тимур ПИ19-2")


def clear():
	my_text.delete(1.0, END)
	my_entry.delete(0, END)


def encrypt():
	# Берем текст из поля
	secret = my_text.get(1.0, END)

	# Очищаем поле
	my_text.delete(1.0, END)

	# Логика для пароля
	# Конвертируем в byte
	secret = secret.encode("ascii")
	# Конвертируем в base64
	secret = pybase64.b64encode(secret)
	# Конвертируем обратно в ascii
	secret = secret.decode("ascii")
	# Вставляем в текстовое поле
	my_entry.insert(END, secret)

	# Работа с записью в файл
	file_object = open('passwords.txt', 'a')
	file_object.write(secret + "\n")
	file_object.close()

	# Сообщение
	messagebox.showwarning("Успех!", "Ваш пароль сохранен в файле passwords.txt")
	# Очистка
	clear()


my_frame = Frame(root)
my_frame.pack(pady=20)

enc_button = Button(my_frame, text="Зашифровать", font=("Helvetica", 18), command=encrypt)
enc_button.grid(row=0, column=0)

clear_button = Button(my_frame, text="Очистить", font=("Helvetica", 18), command=clear)
clear_button.grid(row=0, column=2, padx=10)

enc_label = Label(root, text="Введите ваш пароль на латинице", font=("Helvetica", 14))
enc_label.pack()

my_text = Text(root, width=57, height=10)
my_text.pack(pady=10)


password_label = Label(root, text="Ваш зашифрованный пароль", font=("Helvetica", 14))
password_label.pack()

my_entry = Entry(root, font=("Helvetica", 18), width=35)
my_entry.pack(pady=10)


root.mainloop()
