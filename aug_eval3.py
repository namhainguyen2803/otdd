import csv

file_path = "saved_augmentation_2/acc_dist_method_otdd_exact_maxsize_5000.csv"
with open(file_path, mode ='r')as file:
    csvFile = csv.reader(file)
    list_in4 = list()
    for lines in csvFile:
        list_in4 = lines

print(list_in4[0])
