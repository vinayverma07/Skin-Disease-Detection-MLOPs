import tensorflow as tf

def build_model(input_shape, classes, learning_rate):
    """
    Builds a State-of-the-Art Transfer Learning model using EfficientNetB3.
    Includes regularization layers designed to prevent overfitting on clinical image data.
    """
    # Explicitly use Input layer to resolve the Keras KerasTensor / Sequential warnings
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 1. Base Model: Pre-trained on ImageNet to capture rich edge/texture features
    # Note: Keras EfficientNet has built-in Rescaling, no separate normalization layer needed.
    base_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet', 
        include_top=False, 
        input_tensor=inputs
    )
    
    # Freeze the base model to preserve features during the initial phase
    base_model.trainable = False
    
    # 2. Custom Classification Head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Dense layer for feature consolidation with L2 regularization
    x = tf.keras.layers.Dense(
        256, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    
    # Dropout to heavily combat overfitting on smaller medical image datasets
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile with Adam optimizer using a slightly lower LR for stability in transfer learning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model