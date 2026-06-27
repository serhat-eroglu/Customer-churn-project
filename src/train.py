from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression(X_train, y_train):
    """Initializes and trains a Baseline Logistic Regression Model."""
    model = LogisticRegression(solver='liblinear',
                               max_iter=1000,
                               class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """Initializes and trains a Max-Depth Limited Decision Tree."""
    model = DecisionTreeClassifier(max_depth=5,
                                   random_state=42,
                                   class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Initializes and trains the Hyperparameter-Tuned XGBoost Model."""
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=200,
        scale_pos_weight=2.76,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Evaluates the model and prints a comprehensive classification report."""
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    return y_pred