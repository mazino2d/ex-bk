import tensorflow.keras as keras
import tensorflow.keras.losses as Lo
import tensorflow.keras.metrics as Me
import tensorflow.keras.optimizers as Op
from src.dataset import feeder
from src.netword import cnn_2s

if __name__ == "__main__":
    inpt1, inpt2, oupt = cnn_2s.build()

    model = keras.Model(
        inputs=[inpt1, inpt2],
        outputs=oupt,
    )

    model.compile(
        optimizer=Op.Adam(),
        loss=Lo.BinaryCrossentropy(),
        metrics=[Me.Accuracy()],
    )

    model.summary()

    model.fit(
        x=feeder.build(),
        epochs=4,
        steps_per_epoch=32,
        max_queue_size=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath="tmp/weights.{epoch:02d}.hdf5",
                save_weights_only=True,
                save_freq="epoch",
            ),
            keras.callbacks.TensorBoard(
                log_dir="tmp/logs",
            )
        ]
    )
