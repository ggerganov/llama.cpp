import matplotlib.pyplot as plt
import sys, os

labels = []
numbers = []
numEntries = 0

def bar_chart(numbers, labels, pos):
    plt.bar(pos, numbers, color='blue')
    plt.xticks(ticks=pos, labels=labels)
    plt.title("Jeopardy Results by Model")
    plt.xlabel("Model")
    plt.ylabel("Questions Correct")
    plt.show()

def calculatecorrect():
    directory = os.fsencode("./examples/jeopardy/results/")
    for listing in os.listdir(directory):
        filename = os.fsdecode(listing)
        if filename.endswith(".txt"):
            file = open("./examples/jeopardy/results/" + filename, "rt")
            global labels
            global numEntries
            global numbers
            labels.append(filename[:-4])
            numEntries += 1
            numbers.append(1)



if __name__ == '__main__':
    calculatecorrect()
    pos = list(range(numEntries))
    bar_chart(numbers, labels, pos)
