#%%
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_data = pd.read_csv('labeledTrainData.tsv', sep='\t', dtype={'id': np.string_, 'sentiment': "category", 'review': np.string_})
test_data = pd.read_csv('testData.tsv', sep='\t', dtype={'id': np.string_, 'review': np.string_})

# %% # 檢查是否有nan
print(np.where(pd.isnull(train_data)))
print(np.where(pd.isnull(test_data)))

# %% # 正負評論比例查看
plt.bar(train_data['sentiment'].cat.categories, train_data['sentiment'].value_counts())
# 資料標籤
for a,b in zip(train_data['sentiment'].cat.categories, train_data['sentiment'].value_counts()):  
    plt.text(a, b+100, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
# y軸上限
plt.ylim(0,len(train_data))
# x軸label
plt.xticks(train_data['sentiment'].cat.categories,("positive reviews","negative reviews"))
# x軸標題
plt.xlabel('Sentiment of the review')
# y軸標題
plt.ylabel('Numbers')
plt.savefig('sentiment_bar.png')
plt.show()

# %%
all_data = pd.concat([train_data, test_data], sort=False)
token_list = []
token_size = []

lemma = WordNetLemmatizer()
for text in all_data['review']:
    #轉全小寫
    text = text.lower()

    #刪除數字空白和標點符號
    text = text.replace('"', '')
    text = re.sub("\\\\|\\'|\\<.*?>|\\-|\\.|\\``|\\?|\\!|\\,|\\+|\\=|\\:|\\(|\\)|\\/||\\''", "", text)
    new_text = filter(lambda ch: ch not in '\r\n\t1234567890', text)
    text = ''.join(list(new_text))

     # 斷詞
    tokens = word_tokenize(text, language='english')

    content = []
    # 停用詞列表設置
    stopWords = set(stopwords.words('english'))
    for w in tokens:
        if len(w) > 1:
            # 字根還原
            w = lemma.lemmatize(w)

            # 刪停用詞
            if w not in stopWords:
                content.append(w)
    token_size.append(len(content))
    token_list.append(content)

# %%
all_data['clean_data'] = token_list
all_data.to_csv('all_data.csv', index=0)

# %% # 折線圖
plt.plot(all_data['id'], token_size)
plt.hist(token_size, bins=[i for i in range(0, len(all_data), 100)])

# %%
