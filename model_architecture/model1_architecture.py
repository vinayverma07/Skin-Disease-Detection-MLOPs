import tensorflow as tf
from cnnClassifier import logger
from tensorflow.keras import layers, models

def build_model(self, num_classes: int) -> tf.keras.Model:
        """
        Builds and compiles the DenseNet121 transfer learning architecture.
        """
        logger.info("Building DenseNet121 model architecture...")

        img_shape = tuple(self.config.params_image_size)
        inputs = layers.Input(shape=img_shape)

        # 1. Base Model & Preprocessing
        # Edited
        # x = tf.keras.applications.densenet.preprocess_input(inputs)
        base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            pooling='avg'
        )


        # Fine-tuning: make base model trainable
        base_model.trainable = True

        # 2. Classification Head
        x = base_model.output
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="DenseNet121_BrainCancer")

        # Compile model using categorical_crossentropy for one-hot labels
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary(print_fn=logger.info)
        return model