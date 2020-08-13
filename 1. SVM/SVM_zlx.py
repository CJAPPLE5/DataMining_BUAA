# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import sys
# import imblearn
# from imblearn.under_sampling import RandomUnderSampler

class optStruct:
	"""
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
	"""
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#根据矩阵行数初始化alpha参数为0	
		self.b = 0 										#初始化b参数为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
		self.K = np.mat(np.zeros((self.m,self.m)))		#初始化核K
		for i in range(self.m):							#计算所有数据的核K
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup): 
	"""
	通过核函数将数据转换到更高维的空间
	Parameters：
		X - 数据矩阵
		A - 单个数据的向量
		kTup - 包含核函数信息的元组
	Returns:
	    K - 计算的核K
	"""
	m, n = np.shape(X)
	K = np.mat(np.zeros((m,1)))
	# exchange
	# 线性核函数,只进行内积。
	if kTup[0] == 'lin':
		K = X * A.T
	# 高斯核函数,根据高斯核函数公式进行计算
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j, :] - A
			K[j] = deltaRow*deltaRow.T
		K = np.exp(K/(-2*kTup[1]**2))
	else:
		raise NameError('核函数无法识别')
	return K


def calcEk(oS, k):
	"""
	计算误差
	Parameters：
		oS - 数据结构
		k - 标号为k的数据
	Returns:
	    Ek - 标号为k的数据误差
	"""
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):
	"""
	随机选择alpha_j的索引值
	Parameters:
	    i - alpha_i的索引值
	    m - alpha参数个数
	Returns:
	    j - alpha_j的索引值
	"""
	# 选择一个不等于i的j
	j = i
	while j == i:
		j = int(random.uniform(0, m))
	return j

def selectJ(i, oS, Ei):
	"""
	内循环启发式选择alpha_j的索引值
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
		Ei - 标号为i的数据误差
	Returns:
	    j, maxK - 标号为j或maxK的数据的索引值
	    Ej - 标号为j的数据误差
	"""
	# 初始化
	j = -1; maxDeltaE = 0; Ej = 0
	# 根据Ei更新误差缓存，并找出误差不为0的数据的索引值
	oS.eCache[i] = [1,Ei]
	validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
	# 有不为0的误差，则遍历并找到最优的alpha_j的索引值
	if (len(validEcacheList)) > 1:
		for k in validEcacheList:
			if k == i: continue  # j不能为i
			# 计算Ek 及 |Ei-Ek|，若大于maxDeltaE则更新
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)
			if deltaE > maxDeltaE:
				j = k; maxDeltaE = deltaE; Ej = Ek
		# 返回maxK,Ej
		return j, Ej
	# 没有不为0的误差，则随机选择alpha_j的索引值
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej

def updateEk(oS, k):
	"""
	计算Ek,并更新误差缓存
	Parameters：
		oS - 数据结构
		k - 标号为k的数据的索引值
	Returns:
		无
	"""
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]


def clipAlpha(aj,H,L):
	"""
	修剪alpha_j
	Parameters:
	    aj - alpha_j的值
	    H - alpha上限
	    L - alpha下限
	Returns:
	    aj - 修剪后的alpah_j的值
	"""
	if aj > H: 
		aj = H
	if L > aj:
		aj = L
	return aj

def innerL(i, oS):
	"""
	优化的SMO算法
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
	Returns:
		1 - 有任意一对alpha值发生变化
		0 - 没有任意一对alpha值发生变化或变化太小
	"""
	# 步骤1：计算误差Ei
	Ei = calcEk(oS, i)
	# 优化alpha,设定一定的容错率
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) \
			or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		# 借助启发式方式选择alpha_j，并计算Ej
		j, Ej = selectJ(i, oS, Ei)
		# 保存更新前的aplpha值，使用深拷贝
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		# 步骤2：计算上下界L和H
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H: 
			# print("L==H")
			return 0
		# 步骤3：计算eta
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta >= 0: 
			print("eta>=0")
			return 0
		# 步骤4：更新alpha_j
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
		# 步骤5：修剪alpha_j
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		# 更新Ej至误差缓存
		updateEk(oS, j)
		if abs(oS.alphas[j] - alphaJold) < 0.00001:
			# print("alpha_j变化太小")
			return 0
		# 步骤6：更新alpha_i
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		# 更新Ei至误差缓存
		updateEk(oS, i)
		# 步骤7：更新b_1和b_2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		# 步骤8：根据b_1和b_2更新b
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
	"""
	完整的线性SMO算法
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		maxIter - 最大迭代次数
		kTup - 包含核函数信息的元组
	Returns:
		oS.b - SMO算法计算的b
		oS.alphas - SMO算法计算的alphas
	"""
	# 初始化数据结构
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
	print("-----init finished!-----")
	# 初始化当前迭代次数为0，第一次遍历全样本，初始化alpha改变次数为0
	iter = 0
	entireSet = True
	alphaPairsChanged = 0
	# 若遍历整个数据集alpha未发生更新 或者超过最大迭代次数,则停止迭代
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		iter += 1
		# 遍历整个数据集
		if entireSet:
			for i in range(oS.m):
				# 使用优化的SMO算法
				alphaPairsChanged += innerL(i, oS)
			print("全样本遍历:第%d次迭代, alpha优化次数:%d" % (iter, alphaPairsChanged))
		# 遍历非边界值
		else:
			# 选取非边界值：alpha中在0和C之间的部分
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
			print("非边界遍历:第%d次迭代, alpha优化次数:%d" % (iter, alphaPairsChanged))
		# 全样本遍历一次后，改为非边界遍历
		if entireSet:
			entireSet = False
		# 如果非边界遍历中alpha没有更新,则非边界点全部满足KKT条件，故改为全样本遍历
		elif alphaPairsChanged == 0:
			entireSet = True
	print("-----train finished!-----")
	# 返回SMO算法计算的b和alphas
	return oS.b, oS.alphas


# gamma = 10.0
def train(C, tol, maxIter, k1):
	"""
	测试函数
	Parameters:
		k1 - 使用高斯核函数的时候表示到达率
	Returns:
		无
	"""
	# 根据训练集计算b和alphas
	b, alphas = smoP(trainData, trainLabel, C, tol, maxIter, ('rbf', k1))
	datMat = np.mat(trainData)
	labelMat = np.mat(trainLabel).transpose()
	# 获得支持向量
	svInd = np.nonzero(alphas.A > 0)[0]
	sVs = datMat[svInd]
	labelSV = labelMat[svInd]
	print("支持向量个数:%d" % np.shape(sVs)[0])
	np.save('model/sVs.npy', sVs)
	np.save('model/k1.npy', k1)
	np.save('model/labelSV.npy', labelSV)
	np.save('model/alphas.npy', alphas)
	np.save('model/svInd.npy', svInd)
	np.save('model/b.npy', b)
	return predictAndCalculate(datMat, trainLabel, '训练集', sVs, k1, labelSV, alphas, svInd, b)


def val():
	sVs = np.load('model/sVs.npy')
	k1 = np.load('model/k1.npy')
	labelSV = np.load('model/labelSV.npy')
	alphas = np.load('model/alphas.npy')
	svInd = np.load('model/svInd.npy')
	b = np.load('model/b.npy')
	# testData = dataArr[3200:3250]
	# testLabel = labelArr[3200:3250]  # 加载测试集
	datMat = np.mat(valData)
	labelMat = np.mat(valLabel).transpose()
	return predictAndCalculate(datMat, valLabel, '验证集', sVs, k1, labelSV, alphas, svInd, b)


def test():
	sVs = np.load('model/sVs.npy')
	k1 = np.load('model/k1.npy')
	labelSV = np.load('model/labelSV.npy')
	alphas = np.load('model/alphas.npy')
	svInd = np.load('model/svInd.npy')
	b = np.load('model/b.npy')
	datMat = np.mat(testData)
	m, n = np.shape(datMat)
	results = []
	for i in range(m):
		# 计算各个点的核
		kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
		# 根据支持向量的点，计算超平面，返回预测结果
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		result = np.sign(predict[0, 0])
		results.append(result)
	print(len(test_index_list))
	print(len(results))
	outputDict = {'index': tuple(test_index_list), 'label': tuple(results)}
	outputDict = pd.DataFrame(outputDict)
	outputDict.to_csv('prediction.csv', index=False)
	# predict(testData, testLabel, '测试集', sVs, k1, labelSV, alphas, svInd, b)


def predictAndCalculate(datMat, labelList, str, sVs, k1, labelSV, alphas, svInd, b):
	m, n = np.shape(datMat)
	tp = 0.0
	fp = 0.0
	tn = 0.0
	fn = 0.0
	for i in range(m):
		# 计算各个点的核
		kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
		# 根据支持向量的点，计算超平面，返回预测结果
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		predict = predict[0, 0]
		# 真正例tp：预测为正，实际为正
		if np.sign(predict) > 0 and np.sign(labelList[i]) > 0:
			tp += 1.0
		# 假正例fp：预测为正，实际为负
		if np.sign(predict) > 0 and np.sign(labelList[i]) <= 0:
			fp += 1.0
		# 真负例tn：预测为负，实际为负
		if np.sign(predict) < 0 and np.sign(labelList[i]) < 0:
			tn += 1.0
		# 假负例fn：预测为负，实际为正
		if np.sign(predict) < 0 and np.sign(labelList[i]) >= 0:
			fn += 1.0
	print("%s" % str + "tp:%.2f, tn:%.2f, fp:%.2f, fn:%.2f," % (tp, tn, fp, fn))
	try:
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1 = (2 * precision * recall) / (precision + recall)
		acc = (tp + tn) / (tp + tn + fp + fn)
		print("%s" % str + "正确率: %.2f" % acc)
		print("%s" % str + "f1-score: %.2f" % f1)
	except Exception as e:
		print(e)
		acc = 0.0
		f1 = 0.0
	return tp, tn, fp, fn, acc, f1


def loadDataSet(file, trainSet):
	featureNum = 12
	dataInput = pd.read_csv(file)
	# if trainSet:
		# dataInput = dataInput.sample(frac=1).reset_index(drop=True)
	data = dataInput.iloc[:, 0:featureNum]
	nominal = ['x2', 'x5', 'x6', 'x7', 'x8', 'x9']
	ordinal = ['x4']
	ratio = ['x1', 'x3', 'x12', 'x10', 'x11']
	# value = ordinal + ratio
	# special = ['x10', 'x11']
	dummies = []
	for i in nominal:
		dummies.append(pd.get_dummies(data[i], prefix=i))
	data = data.drop(nominal, axis=1)
	for i in dummies:
		data = pd.concat([data, i], axis=1)
	for i in ratio:
		data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
	'''
	for i in special:
		data[i] = data[i].apply(lambda x: x if x == 0 else 1)
	'''
	data = data.values.tolist()

	if trainSet:
		label = dataInput.iloc[:, featureNum]
		label = label.values.tolist()
		return data, label
	else:
		return data, dataInput['index'].tolist()


if __name__ == '__main__':
	trainStage = False
	if trainStage:
		C_all = [0.1, 1, 5, 10, 20, 50, 100]
		k1_all = [0.1, 1, 5, 10, 20]
		# C_all = [i for i in range(1, 30, 4)]
		# k1_all = [i for i in range(1, 30, 4)]
		# C_all = [10]
		# k1_all = [10]
		train_start = time.clock()
		train_size = 2000
		# val_size = 20000

		tol = 0.001
		maxIter = 100

		dataArr, labelArr = loadDataSet('svm_training_set.csv', trainSet=True)
		'''
        trainData = dataArr[:train_size]
        trainLabel = labelArr[:train_size]
        valData = dataArr[train_size:(train_size + val_size)]
        valLabel = labelArr[train_size:(train_size + val_size)]
        '''
		# trainData, valData, trainLabel, valLabel = train_test_split(dataArr, labelArr, test_size=0.2, random_state=1, stratify=labelArr)
		valData = dataArr
		valLabel = labelArr

		trainData = []
		trainLabel = []
		posCount = 0
		negCount = 0
		for i in range(0, 20000):
			if posCount < train_size / 2 and labelArr[i] == 1:
				posCount += 1
				trainData.append(dataArr[i])
				trainLabel.append(labelArr[i])
			elif negCount < train_size / 2 and labelArr[i] == -1:
				negCount += 1
				trainData.append(dataArr[i])
				trainLabel.append(labelArr[i])
			if posCount == train_size / 2 and negCount == train_size / 2:
				break
		'''
        # RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
        rus = imblearn.RandomUnderSampler(random_state=0)
        trainData, trainLabel = rus.fit_sample(trainData, trainLabel)
        negCount = 0
        posCount = 0
        for i in trainLabel:
            if i > 0:
                posCount += 1
            else:
                negCount += 1
        print('+1: %d' % posCount, '-1: %d' % negCount)
        sys.exit(0)
        '''
		dict = []
		df = pd.DataFrame()

		for C in C_all:
			for k1 in k1_all:
				if C / k1 >= 100: continue
				print('C: %.2f' % C, 'k1: %.2f' % k1)
				templist = [C, k1]
				tp, tn, fp, fn, acc, f1 = train(C, tol, maxIter, k1)
				train_end = time.clock()
				print("训练用时：", (train_end - train_start))
				templist += [tp, tn, fp, fn, acc, f1]

				val_start = time.clock()
				tp, tn, fp, fn, acc, f1 = val()
				val_end = time.clock()
				print("验证用时：", (val_end - val_start))
				templist += [tp, tn, fp, fn, acc, f1]

				dict.append(templist)
				df = pd.DataFrame(np.array(dict))
				df.columns = ['C', 'k1', 't_tp', 't_tn', 't_fp', 't_fn', 't_acc', 't_f1',
							  'v_tp', 'v_tn', 'v_fp', 'v_fn', 'v_acc', 'v_f1']
				# print(df)
		df.to_csv('parameter3.csv')
	else:
		testData, test_index_list = loadDataSet('svm_training_set.csv', trainSet=False)
		# testData = testData[5000:5300]
		test()
