# SVM说明文档

赵梁煊 17373157

## 一、运行方法

main函数中，通过trainStage的取值，可以选择当前的任务类型是训练验证 或是 测试。在train()函数的最后，会将训练好的模型存储到model文件夹下，在test()函数的开始，会将模型读取进来，直接进行预测。

```python
if __name__ == '__main__':
	trainStage = True
	if trainStage:
		# 设置参数
        C = 20
		tol = 0.001
		maxIter = 100
		k1 = 15.0
        # 读入训练数据
		dataArr, labelArr = loadTrainDataSet('svm_training_set.csv')
		# 划分训练集与测试集
        trainData = dataArr[:train_val_bound]
		trainLabel = labelArr[:train_val_bound]
		valData = dataArr[train_val_bound:]
		valLabel = labelArr[train_val_bound:]
		# trainData, testData, trainLabel, testLabel = train_test_split(dataArr, labelArr, test_size=0.2, random_state=1)
		# 开始训练
        train(C, tol, maxIter, k1)
        # 开始验证
		val()
	else:
        # 读入测试数据
		testData = loadTestDataSet('svm_training_set.csv')
		testData = testData[5000:5300]
		# 开始测试
        test()
```

## 二、流程介绍

### 1. 训练及验证

#### （1）数据读入：预处理及特征工程

```python
def loadDataSet(file, trainSet):
	featureNum = 12
	dataInput = pd.read_csv(file)
	if trainSet:
		dataInput = dataInput.sample(frac=1).reset_index(drop=True)
	data = dataInput.iloc[:, 0:featureNum]
	nominal = ['x2', 'x5', 'x6', 'x7', 'x8', 'x9']
	ordinal = ['x4']
	ratio = ['x1', 'x3', 'x12']
	value = ordinal + ratio
	special = ['x10', 'x11']
	dummies = []
	for i in nominal:
		dummies.append(pd.get_dummies(data[i], prefix=i))
	data = data.drop(nominal, axis=1)
	for i in dummies:
		data = pd.concat([data, i], axis=1)
	for i in value:
		data[i] = float(data[i] - data[i].min()) / float(data[i].max() - data[i].min())
	data = data.values.tolist()
    for i in special:
		data[i] = data[i].apply(lambda x: x if x == 0 else 1)
	if trainSet:
		label = dataInput.iloc[:, featureNum]
		label = label.values.tolist()
		return data, label
	else:
		return data
```

训练集和测试集均使用loadDataSet()来读入数据，用trainSet这个参数来表明是否是训练数据。若为训练数据，则进行reindex，且返回data和label；否则是测试数据，不进行reindex，且只返回data。

属性被分为了标称属性nominal、序数属性ordinal、比率属性ratio三种。nominal需要利用pandas的get_dummies函数做变换。ordinal和ratio可以一起利用最大最小值放缩到[0, 1]区域内。此外，注意到x10和x11两个属性的取值，要么是0，要么是非0的较大的数，且0的个数多于非0，再像ratio一样放缩就不合理了，所以把它们按照“是否为非0值”这一标准重新赋值。

#### （2）初始化数据结构

通过读入的训练数据的data和label，以及参数C, toler, gamma进行optStruct的初始化。

```python
class optStruct:
	"""
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		kTup - 包含核函数信息的元组,第一个分量存放核函数类别，第二个分量存放核函数需要用到的参数
	"""
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#初始化alpha为0	
		self.b = 0 										#初始化b为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
		self.K = np.mat(np.zeros((self.m,self.m)))		#初始化核K
		for i in range(self.m):							#计算所有数据的核K
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
```

其中函数kernelTrans的作用是计算所有数据的核K，方便后续利用核函数将数据转换到更高维的空间。当用到的核函数为高斯核函数$K(X_1,X_2)=exp(-\frac{||X_1-X_2||^2}{2\sigma^2})$时，$\sigma$由kTup的第二个分量gamma指示。

```python
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
```

#### （3）SMO算法

```python
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
	"""
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		maxIter - 最大迭代次数
		kTup - 包含核函数信息的元组，第一个分量表示核函数类型（线性，rbf）
		       第二个分量为核函数参数（如对于rbf和函数来说，表示gamma值）
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
	# 若遍历整个数据集alpha未发生更新 或者 超过最大迭代次数,则停止迭代
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		iter += 1
		# 遍历整个数据集
		if entireSet:
			for i in range(oS.m):
				# 使用优化的SMO算法
				alphaPairsChanged += innerL(i,oS)
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
```

其中调用的innerL()函数是SMO()函数的核心，其任务是：对于不满足KKT条件的点，调整alpha的值。

第一步，从不满足KKT条件的alpha点集中选取两个$\alpha_i, \alpha_j$（策略是遍历所有可能的$\alpha_i$，然后根据启发式算法找到对应的最优$\alpha_j$），在调整前后满足$\alpha_i^{new}y_i + \alpha2_2^{new}y_2 =  \alpha_i^{old}y_i + \alpha2_2^{old}y_2 = const$，所以二者知道一个就可以求出另一个。

第二步，我们有alpha的更新公式$\alpha_2^{new}=\alpha_2^{old}-\frac{y_x(E_1-E_2)}{\eta}$，其中$E_I=u_i-y_i, \eta=2K(x_1,x_2)-K(x_1,x_1)-K(x_2,x_2)$，其中$K$为两个向量的核函数值，就可以完成对$\alpha_2^{new}$的更新。但需要注意的是，$\alpha_2^{new}$有上下界限制要求：
$$
L=max(0,\alpha_2^{old}-\alpha_1^{old}), H=min(C,C+\alpha_2^{old}-\alpha_1^{old}).......if\space y_1\neq y_2\\
L=max(0,\alpha_2^{old}+\alpha_1^{old}-C), H=min(C,C+\alpha_2^{old}+\alpha_1^{old})...if\space y_1= y_2
$$
若$\alpha_2^{new}$超出了上下界，需要限制在$[L,H]$的范围之内。

第三步，再根据公式$\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})$更新$\alpha_1$（不需要限制）。

第四步，根据公式计算出$b$的两个候选更新值：
$$
b_1^{new}=b^{old}-E_1-y_1(\alpha_1^{new}-\alpha_1^{old})K(x_1,x_1)-y_2(\alpha_2^{new}-\alpha_2^{old})K(x_1,x_2)\\
b_2^{new}=b^{old}-E_2-y_1(\alpha_1^{new}-\alpha_1^{old})K(x_1,x_2)-y_2(\alpha_2^{new}-\alpha_2^{old})K(x_2,x_2)
$$
根据更新后的$\alpha_1,\alpha_2$选择一个更准确的$b$进行更新：
$$
b=
	\begin{cases}
		b_1^{new}, if \space 0 \leq \alpha_1^{new} \leq C\\
		b_2^{new}, if \space 0 \leq \alpha_2^{new} \leq C\\
		\frac{b_1^{new}+b_2^{new}}{2},others
	\end{cases}
$$

```python
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
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
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
		# 步骤7：更新b1和b2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		# 步骤8：根据b1和b2更新b
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0
```

下面是一些在innerL()函数中用到的函数：

```python
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
```

#### (4)验证及调参

在训练结束后，计算出支持向量并保存下来。（在训练集上也预测了一下，看模型效果如何）

```python
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
    # 保存模型
	np.save('model/sVs.npy', sVs)
	np.save('model/k1.npy', k1)
	np.save('model/labelSV.npy', labelSV)
	np.save('model/alphas.npy', alphas)
	np.save('model/svInd.npy', svInd)
	np.save('model/b.npy', b)
	return predictAndCalculate(datMat, trainLabel, '训练集', sVs, k1, labelSV, alphas, svInd, b)
```

在验证时，将保存的模型读入，然后预测并计算f1-score, accuracy等指标。

```python
def val():
    # 加载模型
	sVs = np.load('model/sVs.npy')
	k1 = np.load('model/k1.npy')
	labelSV = np.load('model/labelSV.npy')
	alphas = np.load('model/alphas.npy')
	svInd = np.load('model/svInd.npy')
	b = np.load('model/b.npy')
	datMat = np.mat(valData)
	labelMat = np.mat(valLabel).transpose()
	return predictAndCalculate(datMat, valLabel, '验证集', sVs, k1, labelSV, alphas, svInd, b)
```

预测并计算的函数predictAndCalculate()如下：

```python
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
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = (2 * precision * recall) / (precision + recall)
	acc = (tp + tn) / (tp + tn + fp + fn)
	print("%s" % str + "正确率: %.2f" % acc)
	print("%s" % str + "f1-score: %.2f" % f1)
	return tp, tn, fp, fn, acc, f1
```

在main中，设定了不同的C和k1(gamma)值，对于每种情况都训练并验证一次，看最终哪个参数组合在一起的效果最好。

```python
if __name__ == '__main__':
	trainStage = True
	if trainStage:
        # 需要调整的参数
		C_all = [i for i in range(1, 20, 3)]
		k1_all = [i for i in range(1, 20, 3)]
        # 固定的参数
		train_size = 1000
		tol = 0.001
		maxIter = 100
		# 加载并划分训练集和验证集（需要保证不同参数有相同的训练集验证集）
        ...
		dict = []
		df = pd.DataFrame()
        # 尝试不同的参数组合
		for C in C_all:
			for k1 in k1_all:
				# 训练并预测
                ...
	else:
		# 测试
        ...
```

### 2. 测试

在main中，通过控制trainStage是否为True来进入训练验证或是测试程序，若trainStage为False，则加载测试集后进入test()函数。

```python
def test():
    # 加载模型
	sVs = np.load('model/sVs.npy')
	k1 = np.load('model/k1.npy')
	labelSV = np.load('model/labelSV.npy')
	alphas = np.load('model/alphas.npy')
	svInd = np.load('model/svInd.npy')
	b = np.load('model/b.npy')
	datMat = np.mat(testData)
	m, n = np.shape(datMat)
	results = []
    # 遍历预测
	for i in range(m):
		# 计算各个点的核
		kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
		# 根据支持向量的点，计算超平面，返回预测结果
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		result = np.sign(predict[0, 0])
		results.append(result)
    # 输出到prediction.csv
	outputDict = {'label': tuple(results)}
	outputDict = pd.DataFrame(outputDict)
	outputDict.to_csv('prediction.csv')
```

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20191229134314358.png" alt="image-20191229134314358"  />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20191229134336319.png" alt="image-20191229134336319"  />

![image-20191229134349456](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20191229134349456.png)