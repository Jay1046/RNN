from model.train import train_model
import settings



train_model(path=settings.settings['path'], epochs=settings.settings['epochs'],)
print("completed")

