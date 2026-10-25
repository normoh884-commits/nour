import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

def build_and_train_model():
    print("📊 Loading MNIST Dataset (Handwritten Digits)...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("🧹 Preprocessing Data (Normalizing Pixels to 0-1)...")
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("🤖 Building the Artificial Neural Network (ANN)...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),          # فرد الصورة 28*28
        Dense(128, activation='relu'),          # الطبقة المخفية
        Dense(10, activation='softmax')         # طبقة الخرج (10 أرقام محتملة)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("🚀 Training the Model (1 Epoch for quick demonstration)...")
    model.fit(x_train, y_train, epochs=1)

    print("\n📈 Evaluating Model...")
    loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
    print(f"\n💡 Model Accuracy on Test Data: {accuracy * 100:.2f}%")
    
    # Testing a single prediction
    print("\n🔍 Testing a single image from the test set...")
    dummy_image = x_test[0:1]
    prediction = model.predict(dummy_image)
    predicted_digit = np.argmax(prediction)
    actual_digit = y_test[0]
    
    print(f"➡️ The model predicted: {predicted_digit}")
    print(f"➡️ The actual label is: {actual_digit}")
    print("Business Note: The Softmax function outputs probabilities for each digit, and we pick the highest one!")

if __name__ == '__main__':
    build_and_train_model()
