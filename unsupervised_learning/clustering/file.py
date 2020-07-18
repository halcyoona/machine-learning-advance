
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeling:
    data = ""
    X = []
    Y = []
    matrix_X=[]
    vectorizer = CountVectorizer()
    lda = LatentDirichletAllocation()


    def __init__(self, file="Movies_TV.txt"):
        corpus = open(file).read()
        self.data = corpus.split('\n')
        self.data.remove(self.data[0])
        self.data.remove(self.data[-1])
        for row in self.data:
            label = row[10:13]
            name = row[16:]
            self.X.append(name)
            self.Y.append(label)

    def countVectorizerTrainer(self):
        self.vectorizer = CountVectorizer()
        self.matrix_X = self.vectorizer.fit_transform(self.X)


    def tfidfVectorizerTrainer(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix_X = self.vectorizer.fit_transform(self.X)


    def ldaTrainer(self,topics):
        self.lda = LatentDirichletAllocation(n_components = topics)
        self.lda.fit(self.matrix_X)



    def printOutput(self, words):
        features = self.vectorizer.get_feature_names()
        words = -1 * words
        for tid, topic in enumerate(self.lda.components_):
            print('topic: ', tid)
            print('wordID: ', topic.argsort()[:words:-1])
            print('word: ', [features[i] for i in topic.argsort()[:words:-1]])
            print('prob: ', [topic[i] for i in topic.argsort()[:words:-1]])

#output with 10 topics and each topic with 15 prominant words
print("-------------")
t1 = TopicModeling()
t1.tfidfVectorizerTrainer()
t1.ldaTrainer(10)
t1.printOutput(15)




#output with 15 topics and each topic with 15 prominant words
print("-------------")
t2 = TopicModeling()
t2.tfidfVectorizerTrainer()
t2.ldaTrainer(15)
t2.printOutput(15)


#output with 20 topics and each topic with 15 prominant words
print("-------------")
t3 = TopicModeling()
t3.tfidfVectorizerTrainer()
t3.ldaTrainer(20)
t3.printOutput(15)
