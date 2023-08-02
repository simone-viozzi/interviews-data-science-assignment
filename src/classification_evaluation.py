from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)

def classification_eval(*, y_pred, y_test):
    display_labels = ["Not looking", "Looking"]

    # Evaluate the model
    classification_rep = classification_report(y_test, y_pred, target_names=display_labels)
    print("Classification Report:\n", classification_rep)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=display_labels)