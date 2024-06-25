from new_solution.bot.EYWA.main import *


counter = 0
right_answers_counter = 0
with open('datasets/data/new.txt', 'r') as new_txt:
    path1, identifier1, vector1 = new_txt.readline().rstrip().split()
    new_txt.seek(0)
    for _ in range(10**6):
        path2, identifier2, vector2 = new_txt.readline().rstrip().split()
        if path1 != path2:
            if identifier1 == identifier2:
                if compare_faces([vector1, vector2]) >= 0.75:
                    right_answers_counter += 1
                counter += 1
                ratio = right_answers_counter / counter
                print(counter, ratio)