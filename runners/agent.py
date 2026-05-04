import json, time
from tqdm.notebook import trange, tqdm
from train import Trainer
#from runners.eval import Inference
from data.preprocessor import load_dataset



# gym wrapper
# reward function
# multi-agent simulation

class Agent:
    def __init__(self, config: dict):
        self.config = config

    
    def __call__(self, run_mode, model_type):

        if run_mode == 'train':
            self.trainer = Trainer(self.config)
            self.train(model_type)
        elif run_mode == 'eval':
            self.evaluator = Inference(self.config)
            self.inference(model_type)
        else:
            raise(f'Unknown run_type parameter - {run_mode}')
    
    
    def train(self, model_type):
        self.trainer.update(model_type, load_dataset)

    def inference(self, model_type):
        self.evaluator.evaluate(model_type, load_dataset)



            
