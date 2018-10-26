


# Demo using pre-trained model
```
python test.py \
    --dataroot ./facades/github \
    --valDataroot ./facades/github \
    --netG ./pre_trained/netG_epoch_9.pth

```


# Demo using pre-trained model to predict rainfall
```
python test_rainfall.py


python test_rainfall.py \
    --dataset pix2pix_val \
    --valDataroot ./facades/github \
    --netG ./pre_trained/netG_epoch_9.pth

```



# Training (Density-aware Deraining network using GT label)
```
python derain_train_2018.py  \
    --dataroot ./facades/DID-MDN-training/Rain_Medium/train2018new \
    --valDataroot ./facades/github \
    --exp ./check \
    --netG ./pre_trained/netG_epoch_9.pth
# Make sure you download the training sample and put in the right folder
```

# Density-estimation Training (rain-density classifier)
```
python train_rain_class.py  \
    --dataset pix2pix_class \
    --dataroot ./facades/DID-MDN-training/Rain_Medium/train2018new  \
    --exp ./check_class



python train_rain_class_zx.py  \
    --dataset pix2pix_class \
    --dataroot ./facades/DID-MDN-training-zx  \
    --exp ./check_class
```
