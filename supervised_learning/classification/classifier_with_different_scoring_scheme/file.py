


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class Learning:
    data = ""
    X = []
    Y = []
    trainX = []
    trainY = []
    testX = []
    testY = []


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

    def countVectorizerTrainer(self, maxFeatures, minDf, maxDf):
        vec = CountVectorizer(max_features=maxFeatures, min_df=minDf, max_df=maxDf)
        matrix_X = vec.fit_transform(self.X)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(matrix_X, self.Y, shuffle = True, train_size = 0.8)


    def tfidfVectorizerTrainer(self, maxFeatures, minDf, maxDf):
        vec = TfidfVectorizer(max_features=maxFeatures, min_df=minDf, max_df=maxDf)
        matrix_X = vec.fit_transform(self.X)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(matrix_X, self.Y, shuffle = True, train_size = 0.8)


    def treeClassifier(self, maxFeatures, maxDepth):
        dtc = DecisionTreeClassifier(max_features=maxFeatures, max_depth=maxDepth)
        dtc.fit(self.trainX, self.trainY)
        labels = dtc.predict(self.testX)
        print('Accuracy Score Tree classifier: ', accuracy_score(self.testY, labels))
        print('Precision Score Tree classifier: ', precision_score(self.testY, labels, average="micro"))
        print('F1 Score Tree classifier: ', f1_score(self.testY, labels, average="micro"))
        print('Recal Score Tree classifier: ', recall_score(self.testY, labels, average="micro"))


    def linearClassifier(self, maxIteration, cpuUse, verbosity):
        lc = SGDClassifier(max_iter =maxIteration, n_jobs=cpuUse, verbose=verbosity)
        lc.fit(self.trainX, self.trainY)
        labels = lc.predict(self.testX)
        print('Accuracy Score linear: ', accuracy_score(self.testY, labels))
        print('Precision Score linear: ', precision_score(self.testY, labels, average="micro"))
        print('F1 Score linear: ', f1_score(self.testY, labels, average="micro"))
        print('Recal Score linear: ', recall_score(self.testY, labels, average="micro"))

    

    def KNClassifier(self, neighbours, algo):
        knn = KNeighborsClassifier(n_neighbors=neighbours, algorithm=algo)
        knn.fit(self.trainX, self.trainY)
        labels = knn.predict(self.testX)
        print('Accuracy Score KNeighbours: ', accuracy_score(self.testY, labels))
        print('Precision Score KNeighbours: ', precision_score(self.testY, labels, average="micro"))
        print('F1 Score KNeighbours: ', f1_score(self.testY, labels, average="micro"))
        print('Recal Score KNeighbours: ', recall_score(self.testY, labels, average="micro"))


    def NBClassifier(self, fitPrior, alph):
        nbc = MultinomialNB(fit_prior=fitPrior, alpha=alph)
        nbc.fit(self.trainX, self.trainY)
        labels = nbc.predict(self.testX)
        print('Accuracy Score MultinomialNB: ', accuracy_score(self.testY, labels))
        print('Precision Score MultinomialNB: ', precision_score(self.testY, labels, average="micro"))
        print('F1 Score MultinomialNB: ', f1_score(self.testY, labels, average="micro"))
        print('Recal Score MultinomialNB: ', recall_score(self.testY, labels, average="micro"))


if __name__ == "__main__":

    print("-------------Tree Classifier scores using count vertorizer---------")
    l1 = Learning()
    l1.countVectorizerTrainer(10, 2, 5)
    l1.treeClassifier(8, 5)
    print("-------------Tree Classifier scores using tfidf----------------")
    l1 = Learning()
    l1.tfidfVectorizerTrainer(10, 2, 5)
    l1.treeClassifier(8, 5)




    print("-------------NB Classifier scores using count vertorizer--------")
    l1 = Learning()
    l1.countVectorizerTrainer(10, 2, 5)
    l1.NBClassifier(True, 5)
    print("-------------NB Classifier scores using tfidf----------------")
    l1 = Learning()
    l1.tfidfVectorizerTrainer(10, 2, 5)
    l1.NBClassifier(True, 5)


    print("-------------Linear Classifier scores using count vertorizer----------")
    l1 = Learning()
    l1.countVectorizerTrainer(10, 2, 5)
    l1.linearClassifier(15,1,0)
    print("-------------Linear Classifier scores using tfidf---------------")
    l1 = Learning()
    l1.tfidfVectorizerTrainer(10, 0, 15)
    l1.linearClassifier(15,1,0)






    print("-------------KNN Classifier scores using count vertorizer----------")
    l1 = Learning()
    l1.countVectorizerTrainer(10, 0, 15)
    l1.KNClassifier(8, "brute")
    print("-------------KNN Classifier scores using tfidf---------------")
    l1 = Learning()
    l1.tfidfVectorizerTrainer(10, 0, 15)
    l1.KNClassifier(8, "brute")




