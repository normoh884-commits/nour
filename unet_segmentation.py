import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_unet(input_shape=(128, 128, 3), num_classes=2):
    '''Builds a simplified U-Net architecture for Image Segmentation.'''
    inputs = layers.Input(shape=input_shape)
    
    # 1. Encoder (Downsampling)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # 2. Bottleneck
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    
    # 3. Decoder (Upsampling) with Skip Connections
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(b)
    concat1 = layers.concatenate([u1, c2]) # Skip connection
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
    
    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c3)
    concat2 = layers.concatenate([u2, c1]) # Skip connection
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)
    
    # Output Layer (Pixel-wise classification)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c4)
    
    return models.Model(inputs=[inputs], outputs=[outputs])

def generate_dummy_satellite_data(num_samples=10, img_size=128, num_classes=2):
    '''Generates synthetic image tensors and segmentation masks to run out-of-the-box.'''
    # Random images (normalized pixels 0 to 1)
    X = np.random.rand(num_samples, img_size, img_size, 3).astype(np.float32)
    # Random masks (classes 0 or 1 for each pixel)
    y = np.random.randint(0, num_classes, size=(num_samples, img_size, img_size)).astype(np.int32)
    return X, y

if __name__ == '__main__':
    print("🏗️ Building U-Net Architecture...")
    model = build_unet()
    model.summary()
    
    print("\n📊 Generating Dummy Satellite Images and Masks...")
    X_train, y_train = generate_dummy_satellite_data()
    X_test, y_test = generate_dummy_satellite_data(num_samples=2)
    
    print("⚙️ Compiling Model...")
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("🚀 Training the Model (Quick demonstration)...")
    model.fit(X_train, y_train, epochs=1, batch_size=2)
    
    print("\n📈 Evaluating Model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n💡 Quick Eval Accuracy: {accuracy * 100:.2f}%")
    print("⚠️ Business Note: For real Satellite Segmentation, we MUST use Intersection over Union (IoU) instead of plain Accuracy to handle class imbalance!")
