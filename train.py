from cfg import MyCFG,CFG
from loss import *
from model import Model
from sklearn import datasets
import jax
import jax.numpy as jnp
from flax.training import train_state
import time
import numpy as np
import optax
class Train:
    a=None
    def __init__(self,cfg:CFG):
        self.cfg=cfg
        self.__put_device()
    def __put_device(self):
        self.cfg.X=jnp.array(self.cfg.X)
        self.cfg.Y=jnp.array(self.cfg.Y)
        self.cfg.X=jax.device_get(self.cfg.X)
        self.cfg.Y=jax.device_get(self.cfg.Y)
    def __get_state(self):
        model=self.cfg.model()
        params=model.init(self.cfg.key,self.cfg.X[:1])['params']
        tx=self.cfg.get_tx()
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    def __get_loss(self,X,Y,params):
        y_pred=self.cfg.model().apply(
            {'params': params}
            ,X)
        return self.cfg.loss_func(Y,y_pred),y_pred
    
    def __val_step(self,state,X,Y):
        loss,y_pred=self.__get_loss(X,Y, state.params)
        score=self.cfg.eval_func(Y,y_pred)
        return loss,score,y_pred

    def __log_every_step(self,epoch,step,batch_num,loss,score,val_loss,val_score,times,d_time):
        print(f"\rEpoch {epoch+1:03d}/{self.cfg.epochs} " ,end="")
        step+=1
        if step!= batch_num:
            print(f"{step}/{batch_num}[{'='*int(step/batch_num*30)}{' '*(30-int(step/batch_num*30))}]",end="")
            sum_time=np.sum(times)
            avg_time=sum_time/step
            after_time=avg_time*(batch_num-step)
            print(f" - ETA:{after_time:.2f}[s]",end="")
            print(f" - loss:{loss:.4f} score:{score:.4f} val_loss:{val_loss:.4f} val_score:{val_score:.4f}",end="")
        else:
            print(f"{step}/{batch_num}[{'='*int(step/batch_num*30)}{' '*(30-int(step/batch_num*30))}]",end="")
            sum_time=np.sum(times)
            avg_time=sum_time/step
            # - 1013s 646ms/step
            print(f" - {sum_time:.0f}s {avg_time*1000:.0f}ms/step",end="")
            print(f" - loss:{loss:.4f} score:{score:.4f} val_loss:{val_loss:.4f} val_score:{val_score:.4f}",end="")
    def train_loop(self, train_X , train_Y , test_X , test_Y):
        @jax.jit
        def __train_step(state,X,Y):
            grads,pred  = jax.grad(self.__get_loss,has_aux=True,argnums=2)(X,Y,state.params)
            state = state.apply_gradients(grads=grads)
            loss = self.cfg.loss_func(Y,pred)
            score = self.cfg.eval_func(Y,pred)
            return state,loss,score
        state=self.__get_state()
        temp=state.params
        x_size=len(train_X)
        batch_num=x_size//self.cfg.batch_size
        perm = jax.random.permutation(self.cfg.key, x_size) # shuffle data
        perms = perm[:batch_num * self.cfg.batch_size]
        perms = perms.reshape((batch_num, self.cfg.batch_size))
        overd_perms = perm[batch_num * self.cfg.batch_size:]
        if len(overd_perms)!=0:
            batch_num+=1
        for epoch in range(self.cfg.epochs):
            times=np.array([])
            for step in range(batch_num):
                if len(overd_perms)!=0 and step==batch_num-1:
                    batch_idx=overd_perms
                else:
                    batch_idx=perms[step]
                temp_time=time.time()
                state,loss,score=__train_step(
                    state,
                    train_X[batch_idx],
                    train_Y[batch_idx]
                )
                val_loss,val_score,y_pred=self.__val_step(state,test_X,test_Y)
                end_time=time.time()
                d_time=end_time-temp_time
                times=np.append(times,d_time)
                self.__log_every_step(epoch,step,batch_num,loss,score,val_loss,val_score,times,d_time)
            print("")
        return state,val_score

    def train(self):
        scores=[]
        i=0
        for train_idx,test_idx in cfg.folds:
            print(f"Fold {i+1}/{cfg.n_fold}")
            train_X,train_Y,test_X,test_Y= self.cfg.X[train_idx],self.cfg.Y[train_idx],self.cfg.X[test_idx],self.cfg.Y[test_idx]
            state,score=self.train_loop(train_X,train_Y,test_X,test_Y)
            scores.append(score)
            i+=1
        print(f"Average CV Score : {np.mean(scores)}")



if __name__ == "__main__":
    a=None
    cfg = MyCFG()
    cfg.batch_size=32
    cfg.epochs=500
    cfg.lr=0.001
    iris=datasets.load_iris()
    X=iris.data
    Y=iris.target
    #one hot encoding
    Y=np.eye(3)[Y]
    cfg.init(loss_func=categorical_crossentropy,eval_func=categorical_crossentropy,model=Model,X=X,Y=Y)
    train=Train(cfg)
    train.train()