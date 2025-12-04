#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An example Intrusion Detection application using Dense, Conv1d and Lstm layers
please cite below works if you find it useful:
Akgun, Devrim, Selman Hizal, and Unal Cavusoglu. "A new DDoS attacks intrusion detection 
model based on deep learning for cybersecurity." Computers & Security 118 (2022): 102748.

Hizal, Selman, Ünal ÇAVUŞOĞLU, and Devrim AKGÜN. "A New Deep Learning Based Intrusion 
Detection System for Cloud Security." 2021 3rd International Congress on Human-Computer 
Interaction, Optimization and Robotic Applications (HORA). IEEE, 2021.
"""

import os
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from models import models_ddos


# ============================================================================
# Configuration
# ============================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='DDoS Intrusion Detection System')
    parser.add_argument('--dataset', type=str, default='Data/ddos_dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--model', type=str, default='conv1d',
                        choices=['conv1d', 'dense', 'lstm'],
                        help='Model architecture to use')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--units', type=int, default=64,
                        help='Number of units/filters in model layers')
    parser.add_argument('--output-dir', type=str, default='./savemodels',
                        help='Directory to save model weights')
    return parser.parse_args()


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def loadDataset(filename, test_size=0.20, val_size=0.125):
    """
    Load and preprocess DDoS dataset.
    
    Args:
        filename (str): Path to CSV file
        test_size (float): Fraction of data for testing (default: 0.20)
        val_size (float): Fraction of training data for validation (default: 0.125)
    
    Returns:
        Tuple of (train_data, train_label, val_data, val_label, test_data, test_label)
    """
    try:
        trainfile = pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")
    
    data = pd.DataFrame(trainfile).to_numpy()
    
    # Remove DrDoS_LDAP class (only 12 classes needed)
    data = data[data[:, 67] != 'DrDoS_LDAP']
    np.random.shuffle(data)
    
    # Extract labels
    label = data[:, 67].astype('str')
    
    # Map labels to class indices
    label_map = {
        'WebDDoS': 0, 'BENIGN': 1, 'UDP-lag': 2, 'DrDoS_NTP': 3,
        'Syn': 4, 'DrDoS_SSDP': 5, 'DrDoS_UDP': 6, 'DrDoS_NetBIOS': 7,
        'DrDoS_MSSQL': 8, 'DrDoS_SNMP': 9, 'TFTP': 10, 'DrDoS_DNS': 11
    }
    
    for key, value in label_map.items():
        label[label == key] = value
    
    # Select features (40 most important features)
    feature_indices = -1 + np.array([
        38, 47, 37, 48, 11, 9, 7, 52, 10, 36, 1, 34, 4, 17, 19, 57, 21,
        18, 22, 24, 32, 50, 23, 55, 51, 5, 3, 39, 40, 43, 58, 12, 25,
        20, 2, 35, 67, 33, 6, 53
    ])
    
    data = data[:, feature_indices]
    
    # MIN-MAX normalization
    dmin = data.min(axis=0)
    dmax = data.max(axis=0)
    data = (data - dmin) / (dmax - dmin + 1e-8)  # Add epsilon to avoid division by zero
    
    # Split: Test 20%, Train 70%, Validation 10%
    train_data, test_data, train_label, test_label = train_test_split(
        data, label, test_size=test_size, stratify=label
    )
    
    train_data, val_data, train_label, val_label = train_test_split(
        train_data, train_label, test_size=val_size, stratify=train_label
    )
    
    return (
        train_data.astype('float32'), train_label.astype('int32'),
        val_data.astype('float32'), val_label.astype('int32'),
        test_data.astype('float32'), test_label.astype('int32')
    )


# ============================================================================
# Main Training and Evaluation
# ============================================================================
def main():
    """Main training and evaluation pipeline."""
    args = parse_args()
    
    # Number of classes
    nclass = 12
    
    print("=" * 70)
    print("DDoS Intrusion Detection System")
    print("=" * 70)
    print(f"Loading dataset from: {args.dataset}")
    
    # Load dataset
    try:
        train_data, train_labelp, val_data, val_labelp, test_data, test_labelp = \
            loadDataset(args.dataset)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Convert labels to categorical
    train_label = to_categorical(train_labelp, nclass)
    val_label = to_categorical(val_labelp, nclass)
    test_label = to_categorical(test_labelp, nclass)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    inshape = train_data.shape[1]
    print(f"Number of features: {inshape}")
    
    # Compute class weights for imbalanced data
    class_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labelp),
        y=train_labelp
    )
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup callbacks
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=1,
        mode='min'
    )
    
    model_checkpoint_path = os.path.join(
        args.output_dir,
        f'model_{args.model}_{{epoch:03d}}_{{val_accuracy:.4f}}.h5'
    )
    
    modelCheckPoint = ModelCheckpoint(
        model_checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    # Build model
    print(f"\nBuilding {args.model.upper()} model...")
    if args.model == 'conv1d':
        # Reshape data for Conv1D
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        model = models_ddos.model_conv1D(lr=args.lr, N=args.units, inshape=inshape)
    elif args.model == 'dense':
        model = models_ddos.model_dense(lr=args.lr, N=args.units, inshape=inshape)
    else:  # lstm
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        model = models_ddos.model_lstm(lr=args.lr, N=args.units, inshape=inshape)
    
    model.summary()
    
    # Train model
    print(f"\nTraining for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        train_data, train_label,
        shuffle=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(val_data, val_label),
        callbacks=[modelCheckPoint, earlyStopping],
        class_weight=class_weights,
        workers=3,
        verbose=1
    )
    
    # Load best model
    model_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.h5')])
    if model_files:
        best_model = model_files[-1]
        print(f"\nLoading best model: {best_model}")
        model.load_weights(os.path.join(args.output_dir, best_model))
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    pred = model.predict(test_data, verbose=0)
    pred_y = pred.argmax(axis=-1)
    
    cm = confusion_matrix(test_labelp.astype('int32'), pred_y)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    class_labels = [
        'WebDDoS', 'BENIGN', 'UDP-lag', 'DrDoS_NTP', 'Syn',
        'DrDoS_SSDP', 'DrDoS_UDP', 'DrDoS_NetBIOS', 'DrDoS_MSSQL',
        'DrDoS_SNMP', 'TFTP', 'DrDoS_DNS'
    ]
    
    print("\nAccuracy ratios for each class:")
    for i, label_name in enumerate(class_labels):
        if np.sum(cm[i, :]) > 0:
            acc = cm[i, i] / np.sum(cm[i, :])
            print(f"  {label_name:16s} = {acc:.4f}")
    
    # Plot confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(14, 12))
    cmo = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    cmo.plot(ax=ax, xticks_rotation=45, cmap='Blues')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=100)
    print(f"\nConfusion matrix plot saved to {args.output_dir}/confusion_matrix.png")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid()
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=100)
    print(f"Training history plot saved to {args.output_dir}/training_history.png")
    
    # Save history data
    history_data = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    np.save(os.path.join(args.output_dir, 'history.npy'), history_data)
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()