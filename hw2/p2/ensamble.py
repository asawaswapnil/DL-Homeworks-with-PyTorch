import csv
import numpy as np
import pandas as pd

# get data
def import_data(filename):
	with open(filename, 'rt') as f:
		reader = csv.reader(f, delimiter=',', skipinitialspace=True)

		lineData = list()
		cols = next(reader)

		for col in cols:
			# create a list in lineData for each column of data
			lineData.append(list())

		for line in reader:
			for i in range(0, len(lineData)):
				lineData[i].append(line[i])
				# Copy the data from the line into the correct columns.
				# lineData[i].append(line[i])

		data = dict()

		for i in range(0, len(cols)):
			# Create each key in the dict with the data in its column.
			data[cols[i]] = lineData[i]

		return data

if _name_ == "_main_":

	files = [
		'classification-test-file/20test_results.csv',
	]
	flag = 0
	for file in files:
		print(file)
		flag += 1
		data = import_data(file)
		df = pd.DataFrame.from_dict(data)
		print(df.mode(axis=1))
		df.mode(axis=1).to_csv('ensamble.csv')