import tensorflow.keras as keras
import tensorflow.keras.callbacks as Ca
import tensorflow.keras.losses as Lo
import tensorflow.keras.metrics as Me
import tensorflow.keras.optimizers as Op
from src.dataset import feeder
from src.network import cnn_2s

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
        epochs=100,
        steps_per_epoch=1000,
        max_queue_size=1024,
        callbacks=[
            Ca.ModelCheckpoint(
                filepath="tmp/weights.{epoch:02d}.hdf5",
                save_weights_only=True,
                save_freq="epoch",
            ),
            Ca.TensorBoard(
                log_dir="tmp/logs",
            )
        ]
    )
