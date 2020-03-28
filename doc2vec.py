#%%
from gensim.models import doc2vec
import pandas as pd

#%%
all_data = pd.read_csv('all_data.csv')

#%% # 建立語料庫
documents = []
for i, sentence in enumerate(all_data['clean_data']):
    documents.append(doc2vec.TaggedDocument(sentence, [i]))

#%% # 模型建立
model = doc2vec.Doc2Vec(documents, min_count=1, window = 15, size = 100, sample=1e-3, negative=5, workers=4)
model.train(documents, total_examples=model.corpus_count, epochs=10)
model.save('doc2vec.model')

# print(len(model.docvecs))