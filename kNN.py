from numpy import *
import operator
import csv
from itertools import islice  
import matplotlib
import matplotlib.pyplot as plt

#从csv文件中导入数据
def loadDataFromCsv(filename, n):
	csv_file = csv.reader(open(filename, 'r'))
	data = []
	labels = []
	count = 0
	for temp in islice(csv_file, 1, None): 
		data.append(list(map(eval, temp[1:])))
		labels.append(temp[0])
		dataSet = array(data)
		count = count+1
		if count == n:
			break
	return dataSet, labels

#将属性归一化
def autoNorm(dataSet):
	minVal = dataSet.min(0)
	maxVal = dataSet.max(0)
	ranges = maxVal - minVal
	normMid = dataSet-tile(minVal, (dataSet.shape[0], 1))
	normFinal = mormMid/tile(ranges, (dataSet.shape[0], 1))
	return normFinal

#测试函数，制造输入集
def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	label = ['A', 'A', 'B', 'B']
	return group, label

#计算距离
def distanceCalculate(inx, iny):
    dimension = len(inx)
    temp = sum(((inx-iny)**2))
    return (temp)**(1/dimension)

#knn分类
def classify(x, dataSet, labels, k):
	#calculation the distance
	#diffMat = (tile(inx, (dataSet.shape[0], 1))-dataSet)**2
	#sumUp = diffMat.sum(axis = 1)
	#distance = sumUp**0.5
	newLabel = []
	for inx in x:
		distance = array([distanceCalculate(inx, temp) for temp in dataSet])
		sorteddistance = distance.argsort()
		#print("sorted distance display" + sorteddistance + "\n")
		classCount = {}
		for i in range(k):
			voteIlable = labels[sorteddistance[i]]
			classCount[voteIlable] = classCount.get(voteIlable, 0)+1
		sortted = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
		newLabel.append(sortted[0][0])
	return newLabel

#计算准确率
def calculateAccurate(label1, label2):
	count = 0
	for i in range(len(label1)):
		if(label1[i] == label2[i]):
			count = count+1
	return count/len(label1)



train = "train.csv"
test = "test.csv"

print('开始导入训练文件：')
count1 = eval(input("输入读取大小："))
Train_dataSet, Train_labels = loadDataFromCsv(train, count1)
print("训练元组条数：%d", len(Train_labels))

print('开始导入测试文件：')
count2 = eval(input("输入读取大小："))
Test_dataSet, Test_labels = loadDataFromCsv(test, count2)
print("测试元组条数：%d", len(Test_labels))

print("开始分类")
k = eval(input("选择一个k值"))
newlabels = classify(Test_dataSet, Train_dataSet, Train_labels, k)

print('开始计算正确率')
accuration = calculateAccurate(Test_labels, newlabels);


print(accuration)

fig = plt.figure()
plt.title('red means pewdiction is wrong')
plt.xlabel('tuples')
plt.ylabel('Guess label')
ax = fig.add_subplot(111)
ax.scatter(range(len(newlabels)), newlabels, c = 'r')
ax.scatter(range(len(Test_labels)), Test_labels, c = 'b')
plt.grid(True)
plt.show()
