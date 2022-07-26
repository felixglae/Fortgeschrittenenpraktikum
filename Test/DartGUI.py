from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image

# create GUI
root = Tk()
root.title("Dart Game")
root.geometry("600x600")
root.iconbitmap("70956.ico")
root.resizable(True, True)
# get background image
my_img = ImageTk.PhotoImage(Image.open("board.jpg"))
# set background
my_background = Label(root, image=my_img)
my_background.place(x=0, y=0)
# set starting score
new_score = 501
new_score_2 = 501


def new_game():  # function to start a new game
    global new_score_2
    global new_score
    global player
    global player_2
    # get both player names
    player_2 = s_player_name.get()
    player = f_player_name.get()
    # set starting score
    new_score = 501
    new_score_2 = 501
    # create Labels for both players
    pointlvl_2 = Label(root, text=f"{player_2} you need {new_score_2} points to finish", font=("helvetica", 11),
                       fg="white", bg="black")
    pointlvl_2.grid(row=5, column=11, columnspan=3)
    pointlvl = Label(root, text=f"{player} you need {new_score} points to finish", font=("helvetica", 11), fg="white",
                     bg="black")
    pointlvl.grid(row=5, column=1, columnspan=3)


"""def choose2():  # function to log in player 2
    global player_label_2
    player_label_2 = Label(root, text=s_player_name.get())
    pointlvl_2 = Label(root, text=f"{s_player_name.get()} you need {new_score_2} points to finish",
                       font=("helvetica", 11), fg="white", bg="black")
    pointlvl_2.grid(row=5, column=11, columnspan=3)
    s_player_name.delete(0, END)


def choose1():  # function to log in player 1
    global player_label
    player_label = Label(root, text=f_player_name.get())
    pointlvl = Label(root, text=f"{f_player_name.get()} you need {new_score} points to finish", font=("helvetica", 11),
                     fg="white", bg="black")
    pointlvl.grid(row=5, column=1, columnspan=3)
    f_player_name.delete(0, END)
"""

def score2():
    global new_score_2
    global points_2
    global result_2
    global winner_box_2
    global player_2
    global player
    global bust_box_2

    if new_score_2 > 0:
        pointlvl_2 = Label(root)
        pointlvl_2.grid_forget()
        points_2 = int(score_field_2.get())
        result_2 = new_score_2 - points_2
        new_score_2 = result_2
        pointlvl_2 = Label(root, text=f"{s_player_name.get()} you need {new_score_2} points to finish",
                           font=("helvetica", 11), fg="white", bg="black")
        pointlvl_2.grid(row=5, column=11, columnspan=3)
    if new_score_2 == 0:
        player_2 = s_player_name.get()
        player = f_player_name.get()
        winner_box_2 = messagebox.showinfo("Results", f"{player_2} wins, {player} lost!")
    if new_score_2 <= 1 and new_score_2 != 0:
        player_2 = s_player_name.get()
        bust_box_2 = messagebox.showinfo("Warning", f"{player_2} has busted")
        new_score_2 = new_score_2 + points_2
        pointlvl_2 = Label(root, text=f"{s_player_name.get()} you need {new_score_2} points to finish",
                           font=("helvetica", 11), fg="white", bg="black")
        pointlvl_2.grid(row=5, column=11, columnspan=3)
    score_field_2.delete(0, END)


def score1():
    global new_score
    global points
    global result
    global winner_box
    global player
    global player_2
    global bust_box

    if new_score > 0:
        pointlvl = Label(root)
        pointlvl.grid_forget()
        points = int(score_field_1.get())
        result = new_score - points
        new_score = result
        pointlvl = Label(root, text=f"{f_player_name.get()} you need {new_score} points to finish",
                         font=("helvetica", 11), fg="white", bg="black")
        pointlvl.grid(row=5, column=1, columnspan=3)
    if new_score == 0:
        player = f_player_name.get()
        player_2 = s_player_name.get()
        winner_box = messagebox.showinfo("Results", f"{player} wins, {player_2} lost!")
    if new_score <= 1 and new_score != 0:
        player = f_player_name.get()
        bust_box = messagebox.showinfo("Warning", f"{player} has busted")
        new_score = new_score + points
        pointlvl = Label(root, text=f"{player} you need {new_score} points to finish", font=("helvetica", 11),
                         fg="white", bg="black")
        pointlvl.grid(row=5, column=1, columnspan=3)
    score_field_1.delete(0, END)


def show_finish():
    global finish_img
    top = Toplevel()
    top.geometry("700x816")
    top.resizable(False, False)
    top.title("Possible Finishes")
    top.iconbitmap("70956.ico")
    finish_img = ImageTk.PhotoImage(Image.open("finish3.png"))
    finish_label = Label(top, image=finish_img)
    finish_label.pack()


def quit_game():
    root.destroy()


f_player_label = Label(root, text="Enter first Player Name", font=("helvetica", 11), fg="white", bg="black")
f_player_label.grid(row=0, column=1, pady=10, columnspan=2)

f_player_name = Entry(root, width=25, borderwidth=5, font=("helvetica", 11), fg="white", bg="black")
f_player_name.grid(row=1, column=1, columnspan=2, padx=10, sticky=N)

pointlvl = Label(root, text=f"You need {new_score} points to finish", font=("helvetica", 11), fg="white", bg="black")
pointlvl.grid(row=5, column=1, columnspan=2)

s_player_label = Label(root, text="Enter second Player Name", font=("helvetica", 11), fg="white", bg="black")
s_player_label.grid(row=0, column=11, pady=10, columnspan=2)

s_player_name = Entry(root, width=25, borderwidth=5, font=("helvetica", 11), fg="white", bg="black")
s_player_name.grid(row=1, column=11, columnspan=2, padx=10, sticky=E)

pointlvl_2 = Label(root, text=f"You need {new_score_2} points to finish", font=("helvetica", 11), fg="white",
                   bg="black")
pointlvl_2.grid(row=5, column=11, columnspan=2)

score_1_label = Label(root, text="Enter your score", font=("helvetica", 11), fg="white", bg="#0B1042")
score_1_label.grid(row=6, column=1, columnspan=2)

score_field_1 = Entry(root, width=25, borderwidth=5, font=("helvetica", 11), fg="white", bg="black")
score_field_1.grid(row=7, column=1, pady=5, columnspan=2)

score_2_label = Label(root, text="Enter your score", fg="white", bg="#341209", font=("helvetica", 11))
score_2_label.grid(row=6, column=11, columnspan=2)

score_field_2 = Entry(root, width=25, borderwidth=5, font=("helvetica", 11), fg="white", bg="black")
score_field_2.grid(row=7, column=11, pady=5, columnspan=2)

submit_btn_1 = Button(root, text="Submit Score", command=score1, fg="white", bg="#15254B", font=("helvetica", 11))
submit_btn_1.grid(row=8, column=1, columnspan=2)

submit_btn_2 = Button(root, text="Submit Score", command=score2, fg="white", bg="#5A2911", font=("helvetica", 11))
submit_btn_2.grid(row=8, column=11, columnspan=2)

finish_btn = Button(root, text="Show Finishes", command=show_finish, fg="white", bg="black", font=("helvetica", 11))
finish_btn.grid(row=20, column=1, columnspan=2)

new_game_btn = Button(root, text="Start new Game", command=new_game, fg="white", bg="#45250C", font=("helvetica", 11))
new_game_btn.grid(row=20, pady=300, column=6, columnspan=2)

exit_btn = Button(root, text="Quit Game", command=quit_game, fg="white", bg="#45250C", font=("helvetica", 11))
exit_btn.grid(row=20, column=11, columnspan=2)

root.mainloop()
