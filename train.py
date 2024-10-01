import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanIoU

# Nested U-Net (U-Net++)
def build_nested_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Define the U-Net architecture here

    model = models.Model(inputs, outputs)
    return model

# Attention U-Net
def build_attention_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Define Attention U-Net architecture here

    model = models.Model(inputs, outputs)
    return model

# Compile and train the model
def compile_and_train(model, X_train, y_train, X_val, y_val, batch_size=16, epochs=50):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MeanIoU(num_classes=2)])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)
    return history
