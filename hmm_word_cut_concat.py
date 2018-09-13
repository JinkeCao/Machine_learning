 # -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:01:26 2018
   ...:
@author: jcao2014
"""
import jieba
import jieba.analyse
from collections import deque
from re import match
jieba.analyse.set_stop_words('stopwords')
jieba.load_userdict('userdicts')
with open('mars_0911') as fin:
    with open('20180911_cutOptimization','a') as fout:
        for line in fin:
            try:
                line =  line.strip().split('\t')
                dvc = line[0]
                text = line[1].strip('["').strip('"]').split('","')
                source = line[1]
                result = []
                for t in text:
                    cutlist = deque(jieba.cut(t, HMM = True))
                    if len(cutlist) > 1:
                        i = 0
                        while i < len(cutlist):
                            if match('.*(搞笑|好笑|笑死|笑喷|哈哈哈|噗|很逗|太逗|真逗|非常逗|考试|期末|备考|复习|测试|模拟考|会考|看书|学习|做题|刷题|卷子|中考|高考|月考|期中|一模|二模|三模|不及格|挂科|挂了|成绩|出分|饥饿|饿|肚子空|肚子叫).*',cutlist[
i]):
                                if  i == 0:
                                    cutlist[0] = cutlist[0] + cutlist[1]
                                    del cutlist[1]
                                elif i == (len(cutlist)-1):
                                    cutlist[i] = cutlist[i - 1] + cutlist[i]
                                    del cutlist[i-1]
                                    break
                                else:
                                    cutlist[i] = cutlist[i-1] + cutlist[i] + cutlist[i+1]
                                    del cutlist[i+1]
                                    del cutlist[i-1]
                            i += 1
                    print(cutlist)
                    result.append(' '.join(cutlist))
                print(line[0],' '.join(result),line[2], sep ='\t', file=fout)
            except:
                continue
