
# Overview of DANet
![](figs/architecture.png)

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

![](figs/result.png)

### Evolution of the activation maps during training on ILSVRC validation set.
![](figs/timeline.png)
