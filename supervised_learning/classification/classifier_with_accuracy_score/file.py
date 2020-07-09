from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class Learning:
    data = ""
    X = []
    Y = []
    trainX = []
    trainY = []
    testX = []
    testY = []


    def __init__(self, file="badges.data"):
        corpus = open(file).read()
        self.data = corpus.split('\n')
        self.data.remove(self.data[0])
        self.data.remove(self.data[-1])
        for row in self.data:
            label = row[:1]
            name = row[2:]
            self.X.append(name)
            self.Y.append(label)

    def countVectorizerTrainer(self, maxFeatures, minDf, maxDf):
        vec = CountVectorizer(max_features=maxFeatures, min_df=minDf, max_df=maxDf)
        matrix_X = vec.fit_transform(self.X)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(matrix_X, self.Y, shuffle = True, train_size = 0.7)


    def tfidfVectorizerTrainer(self, maxFeatures, minDf, maxDf):
        vec = TfidfVectorizer(max_features=maxFeatures, min_df=minDf, max_df=maxDf)
        matrix_X = vec.fit_transform(self.X)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(matrix_X, self.Y, shuffle = True, train_size = 0.7)


    def treeClassifier(self, maxFeatures, maxDepth):
        dtc = DecisionTreeClassifier(max_features=maxFeatures, max_depth=maxDepth)
        dtc.fit(self.trainX, self.trainY)
        labels = dtc.predict(self.testX)
        print('Accuracy Tree classifier: ', accuracy_score(self.testY, labels))


    def linearClassifier(self, maxIteration, cpuUse, verbosity):
        lc = SGDClassifier(max_iter =maxIteration, n_jobs=cpuUse, verbose=verbosity)
        lc.fit(self.trainX, self.trainY)
        labels = lc.predict(self.testX)
        print('Accuracy linear: ', accuracy_score(self.testY, labels))

    

    def KNClassifier(self, neighbours, algo):
        knn = KNeighborsClassifier(n_neighbors=neighbours, algorithm=algo)
        knn.fit(self.trainX, self.trainY)
        labels = knn.predict(self.testX)
        print('Accuracy KNeighbours: ', accuracy_score(self.testY, labels))

    def NBClassifier(self, fitPrior, alph):
        nbc = MultinomialNB(fit_prior=fitPrior, alpha=alph)
        nbc.fit(self.trainX, self.trainY)
        labels = nbc.predict(self.testX)
        print('Accuracy MultinomialNB: ', accuracy_score(self.testY, labels))


if __name__ == "__main__":

print("-------------")
l1 = Learning()
l1.countVectorizerTrainer(10, 2, 5)
l1.treeClassifier(8, 5)
l1.NBClassifier(True, 0)
l1.KNClassifier(8, "brute")
l1.linearClassifier(8, 2, 0)


print("-------------")
l1 = Learning()
l1.tfidfVectorizerTrainer(10, 2, 5)
l1.treeClassifier(8, 5)
l1.NBClassifier(True, 0)
l1.KNClassifier(8, "brute")
l1.linearClassifier(8, 2, 0)

print("-------------")
l1.countVectorizerTrainer(15, 4, 10)
l1.treeClassifier(10, 8)
l1.NBClassifier(False, 2)
l1.KNClassifier(8, "ball_tree")
l1.linearClassifier(6, 1, 0)

print("-------------")
l1.tfidfVectorizerTrainer(15, 5, 10)
l1.treeClassifier(10, 8)
l1.NBClassifier(False, 2)
l1.KNClassifier(8, "ball_tree")
l1.linearClassifier(6, 1, 0)

print("-------------")
l1.countVectorizerTrainer(20, 6, 15)
l1.treeClassifier(10, 8)
l1.NBClassifier(False, 2)
l1.KNClassifier(8, "ball_tree")
l1.linearClassifier(6, 1, 0)


print("-------------")
l1.tfidfVectorizerTrainer(20, 6, 15)
l1.treeClassifier(10, 8)
l1.NBClassifier(False, 2)
l1.KNClassifier(8, "ball_tree")
l1.linearClassifier(6, 1, 0)
