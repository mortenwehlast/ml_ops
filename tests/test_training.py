from src.models.train_model import TrainWorker


def test_loss_positive():
    worker = TrainWorker()
    worker.epochs = 1
    batch_losses = worker.train()

    for loss in batch_losses:
        assert loss >= 0
