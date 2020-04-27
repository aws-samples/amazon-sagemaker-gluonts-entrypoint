import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from gluonts.dataset.common import ListDataset, TrainDatasets, load_datasets
from gluonts.dataset.loader import InferenceDataLoader, TrainDataLoader
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer
from gluonts.transform import AdhocTransform


def _log1p(data):
    data["target"] = np.log1p(data["target"])
    return data


def expm1(_, yhat: np.ndarray):
    print("Before expm1:", yhat.shape, yhat)
    print("After expm1:", yhat.shape, np.expm1(yhat))
    return np.expm1(yhat)


log1p = AdhocTransform(func=_log1p)
# expm1 = AdhocTransform(func=_expm1)


def main(idir: os.PathLike):
    dataset = load_datasets(metadata=idir / "metadata", train=idir / "train", test=idir / "test")

    trainer = Trainer(epochs=10, batch_size=1, num_batches_per_epoch=2)
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=dataset.metadata.prediction_length,
        trainer=trainer,
        cell_type="gru",
        use_feat_static_cat=True,
        cardinality=[2],
    )

    """
    # Doesn't work
    tdl = TrainDataLoader(
        dataset=dataset.train,
        transform=log1p,
        batch_size=trainer.batch_size,
        ctx=trainer.ctx,
        num_batches_per_epoch=trainer.num_batches_per_epoch,
        shuffle_for_training=False,
    )
    # one_batch: Dict[str, Any] = iter(tdl).__next__()
    # print("Content of one (mini)batch...")
    # for k, v in one_batch.items():
    #    print("--------")
    #    print(f"{k}: {type(v)} = {v}")
    # print("--------")

    estimator = DeepAREstimator(
        freq="D", prediction_length=2, trainer=trainer, cell_type="gru", use_feat_static_cat=True, cardinality=[2]
    )
    predictor = estimator.train(training_data=tdl)
    """

    ds = TrainDatasets(
        metadata=dataset.metadata,
        train=ListDataset(dataset.train, freq=dataset.metadata.freq),
        test=ListDataset(dataset.test, freq=dataset.metadata.freq),
    )
    print(ds.train.list_data)
    print(ds.test.list_data)
    for data_entry in ds.train:
        data_entry["target"] = np.log1p(data_entry["target"])
    for data_entry in ds.test:
        data_entry["target"] = np.log1p(data_entry["target"])
    print(ds.train.list_data)
    print(ds.test.list_data)

    predictor = estimator.train(training_data=ds.train, validation_data=ds.test)
    predictor.output_transform = expm1

    predictor.serialize(Path("model"))
    predictor = Predictor.deserialize(Path("model"))
    print("After reload, predictor.output_transform =", predictor.output_transform)
    # NOTE: it's none, so we have to reattach the output_transform. Alternatively, ser & deser the transforms manually.
    predictor.output_transform = expm1
    print("Re-attach: predictor.output_transform =", predictor.output_transform)
    print(predictor)
    for forecast in predictor.predict(ds.test, num_samples=1):
        print(forecast)

    print("Done")


def predict(self, dataset):
    inference_data_loader = InferenceDataLoader(
        dataset, self.input_transform, self.batch_size, ctx=self.ctx, dtype=self.dtype,
    )
    return self.forecast_generator(
        inference_data_loader=inference_data_loader,
        prediction_net=self.prediction_net,
        input_names=self.input_names,
        freq=self.freq,
        output_transform=self.output_transform,
        num_samples=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=".")
    arg = parser.parse_args()
    main(arg.input_dir)
