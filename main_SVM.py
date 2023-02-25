import cv2
import numpy as np
from sklearn.svm import LinearSVC
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

SAMPLE_LIST = ['test/cat.1020.jpg',
               'test/cat.1010.jpg',
               'test/dog.1046.jpg',
               'test/dog.1013.jpg']

def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def classification(image_path: list[str]) -> list[int]:
    y = []
    for image in imagePaths:
        if('cat' in image):
            y.append(0)
        elif('dog' in image):
            y.append(1)
        else:
            y.append(-1)
    return y


if __name__ == '__main__' :
    imagePaths = sorted(list(paths.list_images('train')))
    # print(imagePaths)
    X = [extract_histogram(cv2.imread(image)) for image in imagePaths]
    y = np.array(classification(imagePaths))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 4)
    clf = LinearSVC(random_state = 4, C = 1.47)
    clf.fit(X_train, y_train)
    #print(clf.coef_[0][123])
    Test_sample = np.array(extract_histogram(cv2.imread(SAMPLE_LIST[3]))).reshape(1,-1)
    print(Test_sample.shape)
    print(clf.predict(Test_sample))
    y_predicted = np.array(clf.predict(X_test))
    precision = precision_score(y_test, y_predicted)
    print("Precision = ",precision)
    recall = recall_score(y_test, y_predicted)
    print("Recall = ", recall)
    F1 = metrics.f1_score(y_test, y_predicted, average = 'weighted')
    print('F1 = ', F1)
    


