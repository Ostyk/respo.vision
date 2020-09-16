import torch

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    based on https://github.com/matterport/Mask_RCNN
    """
    
    #Pyramid CNN specific
    N_STACK = 4 # pyramid height
    A_BLOCK_NUMBER = 4 # nuber of RC blocks
    N_CHANNELS = 32 # inner conv channels

    #Data loading specific
    BATCH_SIZE = 16
    SHUFFLE = True
    NUM_WORKERS = 0
    NORMALIZE = True #(check dataset.py for details)
    TRANSFORM = None #TO DO: implement transforms
    #training specific
    EPOCHS = 50
    N_CLASSES = 3
    LR = 0.0001
    SAVE_CP=True #save checkpoint
    SEED = 42
    EPOCH_TO_RESTORE=0
    
    
    
    def display(self):
        """Display Configuration values."""
        print("-"*100)
        print("\nConfigurations:\n")
        print("-"*100)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("-"*100)
        print("\n")