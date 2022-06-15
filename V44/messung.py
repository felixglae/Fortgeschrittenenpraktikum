import random
from datetime import datetime
from datetime import time

t = 3


def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        t -= 1


msg = 0
msg2 = 0
msg3 = 0
msg4 = 0
first_trial = ["Task_1", "Task_2"]
stim_1 = ["1", "2", "8", "9"]
stim_2 = ["a", "b", "y", "z"]
task = 0
i = 0

for i in range(3):
    if task == 0:
        random.shuffle(first_trial)
        if first_trial[0] == "Task_1":
            task = 1
            random.shuffle(stim_1)
            msg = stim_1[0]
            print(msg)
            msg2 = int(msg)


        else:
            task = 2
            random.shuffle(stim_2)
            msg3 = stim_2[0]
            print(msg3)

        start_time = datetime.now()
        answer = input()

        if task == 2 and answer == "o":
            if msg3 == "a":
                msg5 = 1
            elif msg3 == "b":
                msg5 = 2
            elif msg3 == "y":
                msg5 = 3
            else:
                msg5 = 4
            if msg5 > 2:
                print("correct!")
            else:
                print("wrong!")
        if task == 2 and answer == "a":
            if msg3 == "a":
                msg5 = 1
            elif msg3 == "b":
                msg5 = 2
            elif msg3 == "y":
                msg5 = 3
            else:
                msg5 = 4
            if msg5 < 3:
                print("correct!")
            else:
                print("wrong!")
        if task == 1 and answer == "1":
            if msg2 < 5:
                print("correct!")
            else:
                print("wrong!")
        if task == 1 and answer == "2":
            if msg2 > 5:
                print("correct!")
            else:
                print("wrong!")
        end_time = datetime.now()
        try_time = datetime(2022, 5, 22, 0, 0, 3)
        try_time2 = datetime(2022, 5, 22, 0, 0, 0)
        resulttime = try_time - try_time2
        if end_time - start_time > resulttime:
            print("too slow")

    if task == 1:
        a = random.randint(0, 100)
        # print("Task1 " + str(a))
        if a > 75:
            task = 1
            random.shuffle(stim_1)
            msg = stim_1[0]
            print(msg)
            msg2 = int(msg)

        else:
            task = 2
            random.shuffle(stim_2)
            msg3 = stim_2[0]
            print(msg3)

        start_time = datetime.now()
        answer = input()

        if task == 2 and answer == "o":
            if msg3 == "a":
                msg5 = 1
            elif msg3 == "b":
                msg5 = 2
            elif msg3 == "y":
                msg5 = 3
            else:
                msg5 = 4
            if msg5 > 2:
                print("correct!")
            else:
                print("wrong!")
        if task == 2 and answer == "a":
            if msg3 == "a":
                msg5 = 1
            elif msg3 == "b":
                msg5 = 2
            elif msg3 == "y":
                msg5 = 3
            else:
                msg5 = 4
            if msg5 < 3:
                print("correct!")
            else:
                print("wrong!")
        if task == 1 and answer == "1":
            if msg2 < 5:
                print("correct!")
            else:
                print("wrong")
        if task == 1 and answer == "2":
            if msg2 > 5:
                print("correct!")
            else:
                print("wrong!")
        end_time = datetime.now()
        try_time = datetime(2022, 5, 22, 0, 0, 3)
        try_time2 = datetime(2022, 5, 22, 0, 0, 0)
        resulttime = try_time - try_time2
        if end_time - start_time > resulttime:
            print("too slow")

    if task == 2:
        a = random.randint(0, 100)
        # print("Task2 " + str(a))
        if a > 75:
            task = 2
            random.shuffle(stim_2)
            msg3 = stim_2[0]
            print(msg3)

        else:
            task = 1
            random.shuffle(stim_1)
            msg = stim_1[0]
            print(msg)
            msg2 = int(msg)

        start_time = datetime.now()
        answer = input()
        if task == 2 and answer == "o":
            if msg3 == "a":
                msg5 = 1
            elif msg3 == "b":
                msg5 = 2
            elif msg3 == "y":
                msg5 = 3
            else:
                msg5 = 4
            if msg5 > 2:
                print("correct!")
            else:
                print("wrong!")
        if task == 2 and answer == "a":
            if msg3 == "a":
                msg5 = 1
            elif msg3 == "b":
                msg5 = 2
            elif msg3 == "y":
                msg5 = 3
            else:
                msg5 = 4
            if msg5 < 3:
                print("correct!")
            else:
                print("wrong!")
        if task == 1 and answer == "1":
            if msg2 < 5:
                print("correct!")
            else:
                print("wrong!")
        if task == 1 and answer == "2":
            if msg2 > 5:
                print("correct!")
            else:
                print("wrong!")
        end_time = datetime.now()
        try_time = datetime(2022, 5, 22, 0, 0, 3)
        try_time2 = datetime(2022, 5, 22, 0, 0, 0)
        result_time = try_time - try_time2
        if end_time - start_time > result_time:
            print("too slow")
