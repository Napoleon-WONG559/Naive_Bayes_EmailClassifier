import jieba
import os
import re
import pandas as pd
import time
import math

class spamEmailBayes:
    # 获得停用词表
	def getStopWords(self):
		stopList = []
		for line in open("中文停用词表.txt",encoding='utf-8'):
			stopList.append(line[:len(line) - 1])
		return stopList;

    # 获得词典
	def get_word_list_dict(self, content, wordsList, stopList, wordsDict):
        # 分词结果放入res_list
        # jieba.cut是一个生成器，取出一行中的每一个词
		res_list = list(jieba.cut(content))
		for i in res_list:
            # 对于第二、三个and判断条件，意思是若本行不能只有一个空格或啥也没有，否则不加入字表中
			if i not in stopList and i.strip() != '' and i != None:
				if i not in wordsList:
					wordsList.append(i)
					wordsDict[i] = 1
				else:
					wordsDict[i] += 1

    # 分别计算正常邮件和垃圾邮件中每个词的参数
	def para_cal(self, wordsList, wordsDict):
		paraDict = {}
		mi = 0
		d = len(wordsDict)
		for word in wordsDict:
			mi += wordsDict[word]
		for word in wordsList:
			paraDict[word] = (wordsDict[word] + 1)/(mi + d)
		return paraDict

	# 获得数据集中邮件路径
	def get_File_List(self):
		filename_0 = "newindex"
		file_spam = []
		file_ham = []
		file_test = []
		file_all_spam = []
		file_all_ham = []
		filename_0_ch = pd.read_csv(filename_0,encoding="utf-8",sep="\n",header=None)
		filename_0_c = filename_0_ch.values
		trainNum = (filename_0_c.size*7)//10
		testNum = (filename_0_c.size*3)//10
		for count in range(0,trainNum):
			str = filename_0_c[count][0]
			if(str[0] == 's'):
				filename = str[8:]
				file_spam.append(filename)
			elif(str[0]=='h'):
				filename = str[7:]
				file_ham.append(filename)
		for count in range(trainNum,filename_0_c.size):
			str = filename_0_c[count][0]
			if (str[0] == 's'):
				filename = str[8:]
			elif (str[0] == 'h'):
				filename = str[7:]
			file_test.append(filename)
		for count in range(0,filename_0_c.size):
			str = filename_0_c[count][0]
			if (str[0] == 's'):
				filename = str[8:]
				file_all_spam.append(filename)
			elif (str[0] == 'h'):
				filename = str[7:]
				file_all_ham.append(filename)
		return file_spam, file_ham, file_test, file_all_spam, file_all_ham

	# 计算预测结果正确率
	def calAccuracy(self, allFileList_ham, allFileList_spam, testResult):
		rightCount = 0
		errorCount = 0
		for name, catagory in testResult.items():
			if (catagory == 0):
				if(name in allFileList_ham):
					rightCount += 1
				else:
					errorCount += 1
			else:
				if(name in allFileList_spam):
					rightCount += 1
				else:
					errorCount += 1
		return rightCount / (rightCount + errorCount)

	# 优化的清洗方式
	def extract_chinese(self, text):
	    import re
	    content = ''.join(text)
	    re_code = re.compile("\r|\n|\\s|\d")
	    re_punc = re.compile('r[+-―——"！~★☆─◆‖□●█〓，。、；………&“”≡《》：]|-') #remove punctation 
	    urlinfo = re.compile(r'(http|https)://([\w.]*).')#transform a http link into 'httpaddr'
	    strinfo = re.compile(r'[?>\n.,:-_%！!/()（）]|<[^<>]*>')      #remove some special characters
	    emailinfo = re.compile(r'[\w_-]+@[\w_.-]+(|.com)')    #transform email address into 'emailaddr'
	    numinfo = re.compile(r'\d+')
	    dollarinfo = re.compile(r'[$]+')
	    monthinfo = re.compile(r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec')
	    weekinfo = re.compile(r'Mon|Tue|Wed|Thu|Fri|Sat|Sun')

	    s = re_punc.sub('', content)
	    s = strinfo.sub('', s)
	    s = re_code.sub('', s)
	    s = emailinfo.sub(' 邮箱地址', s)
	    s = monthinfo.sub(' 月份', s)
	    s = weekinfo.sub(' 星期', s)
	    s = numinfo.sub(' 数字', s)
	    s = dollarinfo.sub(' 美元', s)
	    return s


# spam类对象
spam = spamEmailBayes()
# 保存词频的词典
spamDict = {}
normDict = {}
testDict = {}
# 正常邮件和垃圾邮件中的每一个词语对应的参数值
normParaDict = {}
spamParaDict = {}
# 保存每封邮件中出现的词
wordsList = []
wordsDict = {}
# 保存预测结果,key为文件名，值为预测类别
testResult = {}
# 计时开始
time_start = time.time()
# 分别获得正常邮件、垃圾邮件及测试文件名称列表
FileList = spam.get_File_List()
spamFileList = FileList[0]
normFileList = FileList[1]
testFileList = FileList[2]
allFileList_spam = FileList[3]
allFileList_ham = FileList[4]
# 获得停用词表，用于对停用词过滤
stopList = spam.getStopWords()
# 获得正常邮件中的词频
for fileName in normFileList:
    for line in open(fileName, encoding='gb2312'):
        # 过滤掉非中文字符
        line = spam.extract_chinese(line)
        # 将每封邮件出现的词保存在wordsList中
        spam.get_word_list_dict(line, wordsList, stopList, wordsDict)
# 保存正常邮件中每一个词的出现次数
normDict = wordsDict.copy()
# 参数计算
normParaDict = spam.para_cal(wordsList, wordsDict)
print('normDict:', len(normDict))

# 获得垃圾邮件中的词频
wordsDict.clear()
wordsList.clear()
for fileName in spamFileList:
    for line in open(fileName, encoding='gb2312'):
        line = spam.extract_chinese(line)
        spam.get_word_list_dict(line, wordsList, stopList, wordsDict)
spamDict = wordsDict.copy()
spamParaDict = spam.para_cal(wordsList, wordsDict)
print('spamDict', len(spamDict))

# 测试邮件
for fileName in testFileList:
    testDict.clear()
    wordsDict.clear()
    wordsList.clear()
    p_norm = 0
    p_spam = 0
    for line in open(fileName, encoding='gb2312'):
        line = spam.extract_chinese(line)
        spam.get_word_list_dict(line, wordsList, stopList, wordsDict)
    testDict = wordsDict.copy()
    for word in wordsList:
        if word in normParaDict:
            p_norm = p_norm + (testDict[word]) * math.log(normParaDict[word])
        else:
            p_norm = p_norm + (testDict[word]) * math.log(0.00000001)
    for word in wordsList:
        if word in spamParaDict:
            p_spam = p_spam + (testDict[word]) * math.log(spamParaDict[word])
        else:
            p_spam = p_spam + (testDict[word]) * math.log(0.00000001)
    if(p_spam > p_norm):
        testResult[fileName] = 1
    else:
        testResult[fileName] = 0
# 计算分类准确率（测试集中文件名低于1000的为正常邮件）
testAccuracy = spam.calAccuracy(allFileList_ham,allFileList_spam,testResult)
print(testAccuracy)
# 计时结束
time_end = time.time()
# 输出运行时间
time_c = time_end - time_start
print('time cost:', time_c, 's')