
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from string import punctuation as punc
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


class TimeTraining:
    labels = []
    review = []
    final_list = []

    def preprocessing(self):
        portter_stemer = PorterStemmer()
        word_lematizer = WordNetLemmatizer()
        stop_word_list =list(stop_words.ENGLISH_STOP_WORDS)
        data = pd.read_csv('IMDB_Dataset.csv')
        self.review = data.iloc[: , [0]] .values
        self.labels = data.iloc[:, [1]].values
        x= self.review.flatten()
        x = x.tolist()
        tempList, stemmingList, processing_list, tempSplit = [], [], [], []
        jn = " "
        for i in x:
            remove_tags=i.replace('<br /><br />','')
            tempList=remove_tags.split(" ")
            removed_stoping_word=[word for word in tempList if word not in stop_word_list ] #Removing stop words.
            remove_punctuation=[word for word in removed_stoping_word if word not in punc]
            for word in remove_punctuation:
                stemmingList.append(word_lematizer.lemmatize(portter_stemer.stem(word),'v')) 
            joinStr=jn.join(stemmingList)
            processing_list.append(joinStr)
            stemmingList=[]
        self.final_list=[]
        # print(processing_list)
        for i in processing_list:
            tempWord=""
            for char in i:
                if char not in punc:
                    tempWord=tempWord+char
            self.final_list.append(tempWord)

    def timeLinearClassifier(self):
        vector= CountVectorizer(lowercase=True).fit_transform(self.final_list)
        train_x, test_x, train_y, test_y = train_test_split (vector, self.labels, shuffle = True, train_size = 0.7)
        start_time=time.time()
        linear=SGDClassifier().fit(train_x,train_y)
        label = linear.predict(test_x)
        acc = accuracy_score(test_y, label)
        final_time=time.time() - start_time
        print("Linear Classifier : \n","Time Taken : %s seconds" % final_time,"\n Accuracy Score : ", acc )


    def timeDecisionTreeClassifier(self):
        vector= CountVectorizer(lowercase=True).fit_transform(self.final_list)
        train_x, test_x, train_y, test_y = train_test_split (vector, self.labels, shuffle = True, train_size = 0.7)
        start_time=time.time()
        dtclf = DecisionTreeClassifier()
        dtclf.fit(train_x, train_y)
        label = dtclf.predict(test_x)
        acc = accuracy_score(test_y, label)
        final_time=time.time() - start_time
        print("Decision Tree Classifier  : \n","Time Taken : %s seconds" % final_time,"\n Accuracy Score : ", acc )



    def timeKNNClassifier(self):
        vector= CountVectorizer(lowercase=True).fit_transform(self.final_list)
        train_x, test_x, train_y, test_y = train_test_split (vector, self.labels, shuffle = True, train_size = 0.7)
        start_time=time.time()
        KNN = KNeighborsClassifier()
        KNN.fit(train_x, train_y)
        label = KNN.predict(test_x)
        acc = accuracy_score(test_y, label)
        final_time=time.time() - start_time
        print("KNN Classifier  : \n","Time Taken : %s seconds" % final_time,"\n Accuracy Score : ", acc )


    def timeNBClassifier(self):
        vector= CountVectorizer(lowercase=True).fit_transform(self.final_list)
        train_x, test_x, train_y, test_y = train_test_split (vector, self.labels, shuffle = True, train_size = 0.7)
        start_time=time.time()
        NBC = MultinomialNB()
        NBC.fit(train_x, train_y)
        label = NBC.predict(test_x)
        acc = accuracy_score(test_y, label)
        final_time=time.time() - start_time
        print("NB Classifier  : \n","Time Taken : %s seconds" % final_time,"\n Accuracy Score : ", acc )



if __name__ == "__main__":
    l1 = TimeTraining()
    l1.preprocessing()
    l1.timeDecisionTreeClassifier()
    l1.timeKNNClassifier()
    l1.timeLinearClassifier()
    l1.timeNBClassifier() 
    
    
