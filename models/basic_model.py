from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling ,RandomFlip, RandomRotation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


class BasicModel(Model):

    def _define_model(self, input_shape, categories_count):

        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),
            RandomFlip("horizontal"),
            RandomRotation(.01),

            layers.Conv2D(23, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.1),

            layers.Flatten(),

            layers.Dense(64, activation='relu'), 
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),


            layers.Dense(categories_count, activation='softmax')
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.0008),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
