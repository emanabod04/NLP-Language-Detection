# %%
import pandas as pd 
import re,string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve,cross_val_score,KFold
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import cohen_kappa_score,silhouette_score,confusion_matrix,classification_report,accuracy_score,plot_confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bias_variance_decomp




# %%
df = pd.read_csv("lang.csv", index_col=0)
print("Read Data Completed")

# %%
le = LabelEncoder()
df["Label"] = le.fit_transform(df.Language)
print("Encode Data Completed")
# %%
# We need some way under sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X = df[['Text','Language']].to_numpy()
y = df.Label.to_numpy()
X_resampled, y_resampled = rus.fit_resample(X, y)
new_df = pd.DataFrame(X_resampled,columns=['Text','Language'])
new_df['Label'] = pd.Series(y_resampled)

print("Sampling Data Completed")

# ###**BOW**

# %%
# vectorizer = ''
# def BOW(df):
#     all_Partitions = list(df["Text"])
    
#     vectorizer = CountVectorizer(analyzer = "word")
#     bow_matrix = vectorizer.fit(all_Partitions)
#     feature_names = vectorizer.vocabulary_
    
#     bow_matrix = vectorizer.transform(all_Partitions)
#     feature_array = bow_matrix.toarray()
    
#     return feature_array

# X_bag_of_words= BOW(new_df)


# %%
# use TSNE to speed up train
from sklearn.manifold import TSNE
# tsne_BOW = TSNE(n_components=2)
# tsne_X_bag_of_words = tsne_BOW.fit_transform(X_bag_of_words)


# %%
# X_train_bow, X_test_bow, y_train_bow, y_test_bow ,lang_train_bow, lang_test_bow= train_test_split(X_bag_of_words, new_df.Label, new_df.Language , test_size = 0.20, random_state=42)

# %%
########## use PCA to speed up train
from sklearn.decomposition import PCA
# pca_BOW = PCA(n_components=500)
# X_bag_of_words_pca = pca_BOW.fit_transform(X_bag_of_words)
# X_train_bow_pca = pca_BOW.transform(X_train_bow)
# X_test_bow_pca = pca_BOW.transform(X_test_bow)


# %% [markdown]
# ###**Doc2Vec**

# %%

def get_Vector_from_Doc2Vec(corpus):
    tokenized_df =[]
    for paragraph in corpus:
        tokens = gensim.utils.simple_preprocess(paragraph)
        tokenized_df.append(tokens)
    documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_df)]
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(documents,dm=1,vector_size=30, min_count=2, epochs=40, workers =5)
    corpus_vectors=[]
    for line in tokenized_df:
        inferred_vector = doc2vec_model.infer_vector(line)
        corpus_vectors.append(inferred_vector)
    return doc2vec_model, corpus_vectors

doc2vec_model , vec = get_Vector_from_Doc2Vec(new_df['Text'])
X_train_doc2vex, X_test_doc2vex, y_train_doc2vex, y_test_doc2vex = train_test_split(vec, new_df.Label , test_size = 0.20, random_state=42)
print("Vector Data Completed")
# %% [markdown]
# #### We will use TSNE with 2 components with doc2vec

# %%

# tsne_VEC = TSNE(n_components=2)
# tsne_X_vec = tsne_VEC.fit_transform(vec)


# %%
# SVC_clf = SVC()
# SVC_clf.fit(X_train_bow_pca, y_train_bow)
# y_pred_bow_SVC = SVC_clf.predict(X_test_bow_pca)


# %% [markdown]
# #### Apply KNN for BOW

# %%
# from sklearn.neighbors import KNeighborsClassifier
# Knn_clf = KNeighborsClassifier(n_neighbors=7)
# Knn_clf.fit(X_train_bow_pca, y_train_bow)
# y_pred_bow_Knn = Knn_clf.predict(X_test_bow_pca)



# %% [markdown]
# #### Apply LR for BOW

# %%
# from sklearn.linear_model import LogisticRegressionCV
# LR_clf = LogisticRegressionCV(cv=10, random_state=0)
# # Train Decision Tree Classifer
# LR_clf.fit(X_train_bow_pca, y_train_bow)
# #Predict the response for test dataset
# y_pred_bow_LR = LR_clf.predict(X_test_bow_pca)


# %%
from sklearn.ensemble import RandomForestClassifier
# RF_clf = RandomForestClassifier()
# RF_clf.fit(X_train_bow, y_train_bow)
# y_pred_bow_RF = RF_clf.predict(X_test_bow)


# %%
# SVC_clf_d = SVC()
# SVC_clf_d.fit(X_train_doc2vex, y_train_doc2vex)
# # y_pred_doc2vec_SVC = SVC_clf_d.predict(X_test_doc2vex)
# print("SVC Train Completed")

# %% [markdown]
# #### Apply KNN for Doc2Vec

# %%
# from sklearn.neighbors import KNeighborsClassifier
# Knn_clf_d = KNeighborsClassifier(n_neighbors=3)
# Knn_clf_d.fit(X_train_doc2vex, y_train_doc2vex)
# y_pred_doc2vex_Knn = Knn_clf_d.predict(X_test_doc2vex)


# %% [markdown]
# #### Apply RF for Doc2Vec

# %%
RF_clf_d = RandomForestClassifier()
RF_clf_d.fit(X_train_doc2vex, y_train_doc2vex)
# y_pred_doc2vec_RF = RF_clf_d.predict(X_test_doc2vex)


# %%
# kmeans_BOW = KMeans(n_clusters=28, init='k-means++', max_iter=30, n_init=10,random_state=0)
# kmeans_BOW.fit(tsne_X_bag_of_words)
# kmeans_pred_BOW = kmeans_BOW.predict(tsne_X_bag_of_words)
# print(kmeans_pred_BOW)


# %%
# cluster_BOW = AgglomerativeClustering(n_clusters=28, affinity='euclidean', linkage='ward')  
# Hirarical_BOW = cluster_BOW.fit_predict(tsne_X_bag_of_words)
# print(Hirarical_BOW)


# %%
# kmeans_DOC2VEC = KMeans(n_clusters=29, init='k-means++', random_state=0)
# kmeans_DOC2VEC.fit(tsne_X_vec)
# kmeans_pred_Doc2Vec = kmeans_DOC2VEC.predict(tsne_X_vec)  
# print(kmeans_pred_Doc2Vec)

# %%
# cluster_DOC2VEC = AgglomerativeClustering(n_clusters=27, affinity='euclidean', linkage='ward')  
# Hirarical_Doc2Vec_5= cluster_DOC2VEC.fit_predict(tsne_X_vec)
# print(Hirarical_Doc2Vec_5)

# %%
# %%
def clean_text(text):
       # removing the numbers
       text = re.sub(r'\d+\s|\s\d+\s|\s\d+$', ' ', text)
       # converting the text to lower case
       text = text.lower()
       #Replace punctuation with whitespaces
       text = re.compile('[%s]' % re.escape(string.punctuation)).sub('', text)
       return text

# %%




