from dataclasses import dataclass,field
import jax
from typing import Optional, Callable
from loss import rmse,accuracy
import optax
import sklearn.model_selection
import random
import os
import numpy as np
@dataclass
class CFG:
    seed:int=42
    epochs:int = 100
    batch_size:int = 30
    loss_func:Optional[Callable] = None
    eval_func:Optional[Callable] = None
    key=jax.random.PRNGKey(seed)
    folds:list[int]=field(default_factory=list)
    n_fold:int=5
    kf=sklearn.model_selection.KFold(n_splits=n_fold,shuffle=True,random_state=seed)
    model=None
    X:object=None
    Y:object=None
    log_every_func:Optional[Callable] = None
    gradient_checkpointing:bool=False
    gradient_accumulation:bool=False
    gradient_accumulation_steps:int=1
    gradient_clipping:bool=False
    debug:bool=False
    wandb:bool=False
    wandb_project_name:str="my_project"
    a=None
    def set_folds(self,X,Y):
        """
        You have to implement this function, if you want to customize folds
        in default, it uses sklearn.model_selection.KFold
        and you can chage it to sklearn.model_selection.StratifiedKFold
        """
        self.folds=[]
        if isinstance(self.kf,sklearn.model_selection.KFold):
            for train_index, test_index in self.kf.split(X):
                self.folds.append((train_index,test_index))
        elif isinstance(self.kf,sklearn.model_selection.StratifiedKFold):
            for train_index, test_index in self.kf.split(X,Y):
                self.folds.append((train_index,test_index))


    def set_seed(self,seed):
        """
        fix seed
        it is only for jax,
        if you want to fix seed, other frameworks, you have to implement this function
        """
        self.seed=seed
        self.key=jax.random.PRNGKey(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def set_skf(self):
        """
        change folds to StratifiedKFold
        """
        self.kf=sklearn.model_selection.StratifiedKFold(n_splits=self.n_fold,shuffle=True,random_state=self.seed)
        if self.folds!=[]:
            # reset folds
            self.set_folds(self.X,self.Y)
    def init(self,loss_func,model,X,Y,eval_func=rmse,seed=42):
        self.loss_func=loss_func
        self.eval_func=eval_func
        self.model=model
        self.set_seed(seed)
        self.X=X
        self.Y=Y
        self.set_folds(X,Y)
    def get_tx(self)->optax.GradientTransformation:
        """ You have to implement this function, every model """
        pass

# Example of cfg.py using adam optimizer
class MyCFG(CFG):
    lr=0.01
    def get_tx(self):
        return optax.sgd(self.lr)
    def set_wandb_key(self):
        os.environ["WANDB_API_KEY"] = key

if __name__ == "__main__":
    #rmseをsklearnで計算する
    from sklearn.metrics import mean_squared_error
    #[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2]
    y=np.array(
     [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2]
    )

    #y_pred=np.array([2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2])
    y_pred=np.array([2]*30)
    print(rmse(y,y_pred))