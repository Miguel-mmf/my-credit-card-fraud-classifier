from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def get_metrics(model, y_val, y_pred, y_train, y_train_pred_proba, best_threshold):
    """
    Calculate and print various evaluation metrics for the model.
    Args:
        model: The trained model.
        y_val: True labels for the validation set.
        y_pred: Predicted labels for the validation set.
        y_train: True labels for the training set.
        y_train_pred_proba: Predicted probabilities for the training set.
        best_threshold: The best threshold used for classification.
    """
    
    print("Classification Report on validation set:")
    print(f'F1 Score on validation set: {f1_score(y_val, y_pred):.4f}')
    print(f'F1 Score on train set: {f1_score(y_train, y_train_pred_proba):.4f}')
    print(f'Precision on validation set: {precision_score(y_val, y_pred):.4f}')
    print(f'Recall on validation set: {recall_score(y_val, y_pred):.4f}')
    print(f'Precision on train set: {precision_score(y_train, y_train_pred_proba):.4f}')
    print(f'Recall on train set: {recall_score(y_train, y_train_pred_proba):.4f}')
    print(f'ROC AUC on validation set: {round(roc_auc_score(y_val, y_pred),4):.4f}')
    print(f'ROC AUC on train set: {round(roc_auc_score(y_train, y_train_pred_proba),4):.4f}')
    print(f'Best threshold: {best_threshold:.4f}')
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
