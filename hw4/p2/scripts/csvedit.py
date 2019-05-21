import csv
import pandas as pd
def usecsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return(your_list)
data =usecsv("../15test_results.csv")
# print(data.type)
final2=[]
for i in range(len(data)):
	j=1
	# print(data[i])
	while data[i][1][j]!='@':
		print(j)
		j+=1
	mainstr=data[i][1][0:j]
	final2.append(mainstr)
print(final2)
pd.DataFrame(data=final2 ).to_csv('outhw4p2.csv',header=False,index=True)
