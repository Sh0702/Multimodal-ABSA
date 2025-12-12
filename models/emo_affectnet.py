import numpy as np
from sklearn.utils import resample, shuffle
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def pad_and_balance_features(X, y, seq_len=10, feature_dim=512):
    """
    Pads features to the required dimension, reshapes into sequences,
    and balances the classes via oversampling.
    """
    if X.shape[1] < feature_dim:
        X_ = np.concatenate([X, np.zeros((X.shape[0], feature_dim - X.shape[1]))], axis=1)
    else:
        X_ = X
    remainder = len(X_) % seq_len
    if remainder != 0:
        needed = seq_len - remainder
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        pad_idxs = [pos_idx[i % len(pos_idx)] if i % 2 == 0 else neg_idx[i % len(neg_idx)] for i in range(needed)]
        X_padded = np.concatenate([X_, X_[pad_idxs]], axis=0)
        y_padded = np.concatenate([y, y[pad_idxs]], axis=0)
    else:
        X_padded = X_
        y_padded = y
    X_seq = X_padded.reshape(-1, seq_len, feature_dim)
    y_seq = y_padded.reshape(-1, seq_len)[:, 0]
    X_pos = X_seq[y_seq == 1.0]
    X_neg = X_seq[y_seq == 0.0]
    y_pos = y_seq[y_seq == 1.0]
    y_neg = y_seq[y_seq == 0.0]
    if len(y_pos) > len(y_neg):
        X_neg_up, y_neg_up = resample(X_neg, y_neg, replace=True, n_samples=len(y_pos), random_state=42)
        X_bal = np.concatenate([X_pos, X_neg_up])
        y_bal = np.concatenate([y_pos, y_neg_up])
    else:
        X_pos_up, y_pos_up = resample(X_pos, y_pos, replace=True, n_samples=len(y_neg), random_state=42)
        X_bal = np.concatenate([X_pos_up, X_neg])
        y_bal = np.concatenate([y_pos_up, y_neg])
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)
    return X_bal, y_bal

def load_and_prepare_data(features_path, labels_path, seq_len=10, feature_dim=512):
    X = np.load(features_path)
    y = np.load(labels_path)
    X_bal, y_bal = pad_and_balance_features(X, y, seq_len=seq_len, feature_dim=feature_dim)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(seq_len=10, feature_dim=512, lstm_units=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, feature_dim)),
        tf.keras.layers.LSTM(lstm_units, return_sequences=False),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        self.validation_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0])
        val_probs = val_pred[:, 1]  # Probability of class 1
        val_pred_labels = tf.argmax(val_pred, axis=1).numpy()
        val_true = self.validation_data[1]
        f1 = f1_score(val_true, val_pred_labels, average='weighted')
        acc = accuracy_score(val_true, val_pred_labels)
        auc = roc_auc_score(val_true, val_probs)
        print(f"\n📊 Epoch {epoch+1} Metrics - F1: {f1:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        logs['val_f1'] = f1
        logs['val_accuracy'] = acc
        logs['val_auc'] = auc
