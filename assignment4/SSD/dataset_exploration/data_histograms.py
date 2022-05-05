import matplotlib.pyplot as plt
import csv
import numpy as np


def clean_up(data, percentage = False) :
    for i in range(len(data)) :
        for j in range(len(data[i])) :
            data[i][j] = float(data[i][j])
            if percentage :
                data[i][j] = (data[i][j]*100)   #multiplied by 100 to get percentages of the image
    return(data)


colors = ['#A6A3B1', '#489BE6', '#5AE24F', '#FCE134', '#000000', '#F03A2F', '#FF9814', '#FF84C8', '#B542E6']
labels = ['total', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider']


## Size, all classes together

file = open('histo_data_size.csv')
csvreader = csv.reader(file)

list_size = []
for row in csvreader:
        list_size.append(row)


list_size = clean_up(list_size, True)
list_size_no_total = list_size[1:]

plt.hist(list_size_no_total, 43, log = False, histtype='bar', stacked=True, color = colors[1:], label = labels[1:])
plt.legend(loc="upper right")
plt.title('Size distribution, linear scale')
plt.xlabel('Percentage of the image taken by the object')
plt.ylabel('Number of objects')
plt.show()


## Size, separating classes

fig, ((h0, h1, h2), (h3, h4, h5), (h6, h7, h8)) = plt.subplots(nrows=3, ncols=3)
h = [h0, h1, h2, h3, h4, h5, h6, h7, h8]

for i in range(9) :
    h[i].hist(list_size[i], 20, log = True, histtype='bar', color = colors[i], label = labels[i])
    h[i].set_title(labels[i])

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.8)
plt.suptitle('Size distribution by class, log scale', size=14)
plt.show()


## Aspect Ratio, all classes together

file = open('histo_data_ars.csv')
csvreader = csv.reader(file)

list_ars = []
for row in csvreader:
        list_ars.append(row)


list_ars = clean_up(list_ars, False)
list_ars_no_total = list_ars[1:]



plt.hist(list_ars_no_total, 30, log = True, histtype='bar', stacked=True, color = colors[1:], label = labels[1:])
plt.legend(loc="upper right")
plt.title('Scaled AR distribution, log scale')
plt.xlabel('Aspect ratio')
plt.ylabel('Number of objects')
plt.show()


## Aspect Ratio, separating classes

fig, ((h0, h1, h2), (h3, h4, h5), (h6, h7, h8)) = plt.subplots(nrows=3, ncols=3)
h = [h0, h1, h2, h3, h4, h5, h6, h7, h8]

for i in range(9) :
    h[i].hist(list_ars[i], 20, log = True, histtype='bar', color = colors[i], label = labels[i])
    h[i].set_title(labels[i])

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.8)
plt.suptitle('Scaled AR distribution by class, log scale', size=14)
plt.show()

