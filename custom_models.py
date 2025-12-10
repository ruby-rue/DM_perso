"""
Advanced Emotion Recognition Model Training
Train your own high-accuracy emotion detection model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class EmotionModelTrainer:
    def __init__(self, img_size=48, num_classes=7):
        """
        Initialize the emotion model trainer
        
        Args:
            img_size: Input image size (default 48x48)
            num_classes: Number of emotion classes (default 7)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.model = None
        self.history = None
        
        print("="*70)
        print("EMOTION RECOGNITION MODEL TRAINER")
        print("="*70)
        print(f"Image Size: {img_size}x{img_size}")
        print(f"Number of Classes: {num_classes}")
        print(f"Emotions: {', '.join(self.emotion_labels)}")
        print("="*70)
    
    def create_model_v1(self):
        """Create standard CNN model (good baseline)"""
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print("\n✓ Model V1 (Standard CNN) created successfully!")
        return model
    
    def create_model_v2_deep(self):
        """Create deeper CNN model (better accuracy, slower)"""
        model = Sequential([
            # Block 1
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            
            # Block 2
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            
            # Block 3
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            
            # Block 4
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.4),
            
            # Dense layers
            Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print("\n✓ Model V2 (Deep CNN) created successfully!")
        return model
    
    def create_model_v3_efficient(self):
        """Create efficient model (fast training, good for testing)"""
        model = Sequential([
            # Lightweight architecture
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print("\n✓ Model V3 (Efficient) created successfully!")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n✓ Model compiled with learning rate: {learning_rate}")
        print(f"\nModel Summary:")
        print("-"*70)
        self.model.summary()
        print("-"*70)
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=64):
        """
        Prepare data generators with strong augmentation
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            batch_size: Batch size for training
        """
        print("\n" + "="*70)
        print("PREPARING DATA GENERATORS")
        print("="*70)
        
        # Strong data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            zoom_range=0.3,
            shear_range=0.3,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\n✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Class indices: {train_generator.class_indices}")
        print("="*70)
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=100, 
              model_name='emotion_model_best.h5'):
        """
        Train the model with callbacks
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            model_name: Name for saving the best model
        """
        print("\n" + "="*70)
        print(f"STARTING TRAINING - {epochs} EPOCHS")
        print("="*70)
        
        # Create logs directory
        log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            model_name,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        
        print("\nCallbacks configured:")
        print("✓ Model checkpoint (saves best model)")
        print("✓ Early stopping (patience: 15 epochs)")
        print("✓ Learning rate reduction (patience: 5 epochs)")
        print("✓ TensorBoard logging")
        print("\nTraining started...\n")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard],
            verbose=1
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
        print(f"✓ Best model saved to: {model_name}")
        print(f"✓ TensorBoard logs saved to: {log_dir}")
        print(f"\nTo view training progress:")
        print(f"  tensorboard --logdir={log_dir}")
        print("="*70)
        
        return self.history
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot and save training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to: {save_path}")
        plt.show()
    
    def evaluate(self, test_generator):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("EVALUATING MODEL ON TEST SET")
        print("="*70)
        
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        
        print(f"\n✓ Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"✓ Test Loss: {test_loss:.4f}")
        print("="*70)
        
        return test_accuracy, test_loss
    
    def save_model_for_production(self, model_path='emotion_model_production.h5'):
        """Save model in production-ready format"""
        self.model.save(model_path)
        print(f"\n✓ Production model saved to: {model_path}")
        
        # Also save as TFLite for mobile
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_path = model_path.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TFLite model saved to: {tflite_path}")

def main():
    """Main training function"""
    print("\n")
    print("="*70)
    print(" "*15 + "EMOTION RECOGNITION MODEL TRAINING")
    print("="*70)
    
    # ==================== CONFIGURATION ====================
    # Dataset paths
    TRAIN_DIR = 'data/fer2013/train'
    VAL_DIR = 'data/fer2013/test'
    TEST_DIR = 'data/fer2013/test'
    
    # Training parameters
    MODEL_VERSION = 'v2'  # Options: 'v1', 'v2', 'v3'
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = f'emotion_model_{MODEL_VERSION}_best.h5'
    
    print("\nConfiguration:")
    print("-"*70)
    print(f"Training Directory:   {TRAIN_DIR}")
    print(f"Validation Directory: {VAL_DIR}")
    print(f"Test Directory:       {TEST_DIR}")
    print(f"Model Version:        {MODEL_VERSION}")
    print(f"Epochs:               {EPOCHS}")
    print(f"Batch Size:           {BATCH_SIZE}")
    print(f"Learning Rate:        {LEARNING_RATE}")
    print(f"Save Path:            {MODEL_SAVE_PATH}")
    print("-"*70)
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ ERROR: Training directory not found: {TRAIN_DIR}")
        print("\nPlease organize your dataset as follows:")
        print("data/")
        print("  fer2013/")
        print("    train/")
        print("      angry/")
        print("      disgust/")
        print("      fear/")
        print("      happy/")
        print("      sad/")
        print("      surprise/")
        print("      neutral/")
        print("    test/")
        print("      (same structure)")
        print("\nDownload FER-2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    # Create trainer
    trainer = EmotionModelTrainer(img_size=48, num_classes=7)
    
    # Select and create model
    print("\nCreating model architecture...")
    if MODEL_VERSION == 'v1':
        trainer.create_model_v1()
    elif MODEL_VERSION == 'v2':
        trainer.create_model_v2_deep()
    elif MODEL_VERSION == 'v3':
        trainer.create_model_v3_efficient()
    else:
        print(f"Unknown model version: {MODEL_VERSION}")
        return
    
    # Compile model
    trainer.compile_model(learning_rate=LEARNING_RATE)
    
    # Prepare data
    train_gen, val_gen = trainer.prepare_data_generators(
        TRAIN_DIR,
        VAL_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Train model
    input("\nPress ENTER to start training (or Ctrl+C to cancel)...")
    history = trainer.train(
        train_gen,
        val_gen,
        epochs=EPOCHS,
        model_name=MODEL_SAVE_PATH
    )
    
    # Plot training history
    trainer.plot_training_history(f'training_history_{MODEL_VERSION}.png')
    
    # Evaluate on test set
    if os.path.exists(TEST_DIR):
        print("\nPreparing test data...")
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(48, 48),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        trainer.evaluate(test_gen)
    
    # Save production model
    trainer.save_model_for_production(f'emotion_model_{MODEL_VERSION}_production.h5')
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*70)
    print(f"\nYour trained model is ready to use!")
    print(f"Model file: {MODEL_SAVE_PATH}")
    print(f"\nNext steps:")
    print("1. Use this model in the desktop app")
    print("2. Test it with: python test_model.py")
    print("3. Deploy it in production")
    print("\nHappy emotion detecting! 😊")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()