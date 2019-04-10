
# Overview of DANet
![](figs/fig1-1.png)

# Train
We finetune the DANet model on the ILSVRC dataset.  
```
cd scripts
sh train_DA_cub.sh
```
# Test
```
cd scripts
sh val_DA_cub.sh
```

![](figs/imagenet-box-1.png)

### Evolution of the activation maps during training on ILSVRC validation set.
![](figs/show1-1.png)
