#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import csv

labels = []
numbers = []
numEntries = 1

rows = []


def bar_chart(numbers, labels, pos):
    plt.bar(pos, numbers, color='blue')
    plt.xticks(ticks=pos, labels=labels)
    plt.title("Jeopardy Results by Model")
    plt.xlabel("Model")
    plt.ylabel("Questions Correct")
    plt.show()


def calculatecorrect():
    directory = os.fsencode("./examples/jeopardy/results/")
    csv_reader = csv.reader(open("./examples/jeopardy/qasheet.csv", 'rt'), delimiter=',')
    for row in csv_reader:
        global rows
        rows.append(row)
    for listing in os.listdir(directory):
        filename = os.fsdecode(listing)
        if filename.endswith(".txt"):
            file = open("./examples/jeopardy/results/" + filename, "rt")
            global labels
            global numEntries
            global numbers
            labels.append(filename[:-4])
            numEntries += 1
            i = 1
            totalcorrect = 0
            for line in file.readlines():
                if line.strip() != "------":
                    print(line)
                else:
                    print("Correct answer: " + rows[i][2] + "\n")
                    i += 1
                    print("Did the AI get the question right? (y/n)")
                    if input() == "y":
                        totalcorrect += 1
            numbers.append(totalcorrect)


if __name__ == '__main__':
    calculatecorrect()
    pos = list(range(numEntries))
    labels.append("Human")
    numbers.append(48.11)
    bar_chart(numbers, labels, pos)
    print(labels)
    print(numbers)
