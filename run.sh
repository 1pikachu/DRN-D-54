cp /home2/pytorch-broad-models/DRN-D-54/drn_d_54-0e0534ff.pth . 


python classify.py test --arch drn_d_54 --dummy --pretrained --num_iter 200 --num_warmup 20 -b 1 --jit --nv_fuser --bn_folding  --precision float16 --device cuda  --profile
