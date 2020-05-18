# junn
deep learning framework build around pytorch for personal projects


## Install

```
pip install git+https://github.com/jutanke/junn.git
```
or locally by
```
python setup.py install
```

## Usage

### Scaffolding
```python
from junn.scaffolding import Scaffolding


class MyModel(Scaffolding):

    def __init__(self, model_seed=0,
                 force_new_training=False):
        """
        :param model_seed: {int} random seed for training. Allows for training the same model
                        with different seeds
        :param force_new_training: {boolean} if True delete existing training data
        """
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)
         
        self.model = torch....

    def get_unique_directory(self):
        """ return {string} that uniquely identifies this model. The model seed will be added
            to the string automatically and does not need to be added in this function
        """
        return "mymodel"
    
    def forward(self, x):
        """ pytorch forward function
        """
        return self.model(x)

```
