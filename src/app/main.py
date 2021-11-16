import tensorflow.keras as keras
import tensorflow.keras.losses as Lo
import tensorflow.keras.metrics as Me
import tensorflow.keras.optimizers as Op
from src.netword import cnn_v1

if __name__ == "__main__":
    inpt1, inpt2, oupt = cnn_v1.build()

    model = keras.Model(
        inputs=[inpt1, inpt2],
        outputs=oupt,
    )

    model.compile(
        optimizer=Op.Adam(),
        loss=Lo.BinaryCrossentropy(),
        metrics=[Me.AUC()],
    )

    model.summary()
