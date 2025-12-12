import argparse
import datetime
import tensorflow as tf
from models.emo_affectnet import load_and_prepare_data, build_model, MetricsCallback

def main():
    parser = argparse.ArgumentParser(description="Run EMO AffectNet LSTM experiment.")
    parser.add_argument('--features_path', type=str, required=True, help='Path to X_features.npy')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to y_labels.npy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--lstm_units', type=int, default=128)
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        args.features_path, args.labels_path, seq_len=args.seq_len, feature_dim=args.feature_dim
    )
    model = build_model(seq_len=args.seq_len, feature_dim=args.feature_dim, lstm_units=args.lstm_units)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    metrics_cb = MetricsCallback(val_data=(X_val, y_val))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tensorboard_cb, metrics_cb]
    )
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    main()
