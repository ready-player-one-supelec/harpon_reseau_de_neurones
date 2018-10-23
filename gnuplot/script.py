import numpy

with open("data.data", "r") as file:
    output = []
    row = file.readline().split(";")
    for i in range(1, len(row)):
        output.append(row[i].strip("\n") + " ")
    for row in file.readlines():
        row = row.split(";")
        for i in range(1, len(row)):
            output[i-1] += str(float(row[i].strip("\n"))) + " "

with open("test.data", "w") as file:
    file.write('\n'.join(output))

# 2
with open("test.data", "r") as file:
    output = [[float(value) for value in row.split(" ")[:-1]] for row in file]
    for i in range(len(output)):
        output[i] = [str(output[i][0]), str(numpy.mean(output[i][1:])), str(numpy.std(output[i][1:]))]
    output = [' '.join(row) for row in output]

with open('test2.data', "w") as file:
    file.write("iterations erreur-moyenne ecart-type\n")
    file.write('\n'.join(output))
