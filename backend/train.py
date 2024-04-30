from backend.utils import load_data, clean_data, feature_engineer, create_test_train

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn import preprocessing

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# load data
pickle_raw = load_data('backend/data/Pickleball - Sheet1.csv')

# clean data
pickle_clean = clean_data(pickle_raw)

# create features
pickle = feature_engineer(pickle_clean)

# create test/train data
X_train, X_test, y_train, y_test = create_test_train(pickle)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                            #    feature_names=pickle.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
