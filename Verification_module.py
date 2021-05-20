import pickle

file = open("Models\\Names_Class_Mapping.pickle",'rb')
name_mapping = pickle.load(file)

def main(query, pred):
    
    Person_Name = name_mapping[pred[0]]
    filename = open("Models\\One_Class_Models\\"+str(Person_Name)+"_Model.pickle", "rb")
    clf = pickle.load(filename)
    predicted = clf.predict(query)
    print(Person_Name, predicted)
    if predicted[0] == -1:
        return "Unknown"
    else:
        return name_mapping[pred[0]]