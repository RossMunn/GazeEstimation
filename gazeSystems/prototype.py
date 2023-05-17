import tkinter as tk

class Keyboard(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry('1200x600')  # Adjust as needed.
        self.grid()
        self.selected_key = None
        self.create_widgets()
        self.focus_set()
        self.bind("<Left>", self.move_left)
        self.bind("<Right>", self.move_right)
        self.bind("<Up>", self.move_up)
        self.bind("<Down>", self.move_down)

    def create_widgets(self):
        self.buttons = []
        keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(4):  # Adjust as needed for number of rows.
            row = []
            for j in range(7):  # Adjust as needed for keys per row.
                if 7*i+j < len(keys):
                    button = tk.Button(self, text=keys[7*i+j], height=8, width=14)
                    button.grid(row=i, column=j)
                    row.append(button)
            self.buttons.append(row)

        # Default select first key
        self.select_key(self.buttons[0][0])

    def select_key(self, button):
        if self.selected_key:
            self.selected_key.config(bg="SystemButtonFace")
        button.config(bg="green")
        self.selected_key = button

    def move_left(self, event):
        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                if button == self.selected_key:
                    if j > 0:
                        self.select_key(self.buttons[i][j-1])
                    else:
                        self.select_key(self.buttons[i][-1])  # Go to last element in row
                    return

    def move_right(self, event):
        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                if button == self.selected_key:
                    if j < len(row) - 1:
                        self.select_key(self.buttons[i][j+1])
                    else:
                        self.select_key(self.buttons[i][0])  # Go to first element in row
                    return

    def move_up(self, event):
        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                if button == self.selected_key:
                    if i > 0:
                        self.select_key(self.buttons[i-1][j])
                    else:
                        self.select_key(self.buttons[-1][j])  # Go to last element in column
                    return

    def move_down(self, event):
        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                if button == self.selected_key:
                    if i < len(self.buttons) - 1:
                        self.select_key(self.buttons[i+1][j])
                    else:
                        self.select_key(self.buttons[0][j])  # Go to first element in column
                    return

root = tk.Tk()
app = Keyboard(master=root)
app.mainloop()




