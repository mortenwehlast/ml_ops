import pytest
from src.models.train_model import TrainWorker


def test_loss_positive():
    worker = TrainWorker()
    worker.epochs = 1
    batch_losses = worker.train()

    for loss in batch_losses:
        assert loss >= 0


def test_raises_on_plot_before_run():
    with pytest.raises(ValueError):
        worker = TrainWorker()
        worker.make_plots()
