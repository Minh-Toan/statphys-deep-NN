import csv

# Define the header
header = ['alpha', 'mmse', 'qw', 'q2']

# Path to the CSV file
info_path = "info.csv"
unfo_path = "unfo.csv"

# Create the file and write the header
with open(info_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

with open(unfo_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)