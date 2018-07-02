from pandas import DataFrame, read_table, ExcelWriter
from app_encoder import list2str
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from time import sleep
from json import loads

def list2str(x):
    appstr = ''
    if x != 'null':
        x = loads(x)
        for i in x:
            if i['load_info']!=None:
                n = len(i['load_info'])
                for j in range(n + 1):
                    appstr += (i['app_name'] + ' ')
    return appstr

def app_chi2(src, df1):
    if src == 'usage': 
        df1 = df1[(df1['source1']=='ime_app') | (df1['source1']=='sdk_log') | (df1['source1']=='vcoam_log')]
    else: 
        df1 = df1[(df1['source1']=='ime_app_install') | (df1['source1']=='sdk_log_install')]
    model = CountVectorizer()
    x_vec = model.fit_transform(df1['app_info'])
    ll = x_vec.toarray()
    c = model.get_feature_names()
    x_train = DataFrame(data=ll, columns=c)
    print(x_train.shape)
    model = SelectKBest(chi2, k=2)
    model.fit_transform(x_train, df1['type'])
    df2 = DataFrame.from_dict({c: p for c, p in zip(c, model.pvalues_) if p < 0.05}, orient='index')
    df2.rename(columns={0: "p_val"}, inplace=True)
    df2.sort_values(by='p_val', inplace = True)
    return df2

df1 = read_table('qianzhan_app_0514.txt', header=None, names=['dvc', 'collect_day', 'app_info', 'type', 'source1', 'source2'])
df1.fillna(int(0), inplace=True)
df1.drop_duplicates(["dvc"], inplace=True)
df1['type'], df1['app_info'] = df1.type.apply(int), df1.app_info.apply(list2str)


writer = ExcelWriter('qianzhan_TopApp_20180521.xlsx')
app_chi2(src='install', df1=df1).to_excel(writer,'install')
app_chi2(src='usage', df1=df1).to_excel(writer,'usage')
writer.save()
