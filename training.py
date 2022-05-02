import pickle
import argparse
from sklearn import svm
from sklearn import svm, metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from src.dataloader import DataLoader


def training(x_train, x_test, y_train, y_test):
    
    model = svm.SVC(kernel="rbf", gamma=1, class_weight={0:4,1:2})
    print("--------------WAIT FOR MODEL TRAINING-------------")
    model.fit(x_train, y_train)
    predicted_train = model.predict(x_train)
    predicted_test = model.predict(x_test)
    cm_test = confusion_matrix(y_test, predicted_test)
    cm_train = confusion_matrix(y_train, predicted_train)
    print("Accuracy on testing: ", metrics.accuracy_score(y_test, predicted_test)*100)
    print("Accuracy on training:", metrics.accuracy_score(y_train, predicted_train)*100)
    print("Confuse matrix on testing\n", cm_test)
    print("Confuse matrix on training\n", cm_train)
    pickle.dump(model, open("./models/SVM_AntiSpoof_Face.pkl", "wb"))


def parser_arumment():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r","--real", required=True, help="Path images real")
    ap.add_argument("-f","--fake", required=True, help="Path images fake")
    args = vars(ap.parse_args())

    return args

if __name__ == "__main__":
    args = parser_arumment()
    dataloader = DataLoader(args["real"], args["fake"])
    x_train, x_test, y_train, y_test = dataloader.process()
    training(x_train, x_test, y_train, y_test)