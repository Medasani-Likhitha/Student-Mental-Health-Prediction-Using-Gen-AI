from sklearn.metrics import accuracy_score, f1_score
import evaluate

accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)

rouge = evaluate.load("rouge")


