
# Overview of DANet
![](figs/architecture_v10.pdf)

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

![](figs/imagenet_result1.pdf)

### Evolution of the activation maps during training on ILSVRC validation set.
![](figs/timeline1.pdf)
