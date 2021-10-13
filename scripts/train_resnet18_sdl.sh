################### Training Single Domain Learning Networks ###################
function train_fn {
    CUDA_VISIBLE_DEVICES=<gpu id> python train_net.py --model.dir ./saved_results/sdl --model.name=$1 --data.train $2 --data.val $2 --data.test $2 --train.batch_size=$3 --train.learning_rate=$4 --train.max_iter=$5 --train.cosine_anneal_freq=$6 --train.eval_freq=$6
}

# Train an single domain learning network on every training dataset (the following models could be trained in parallel)

# ImageNet
NAME="imagenet-net"; TRAINSET="ilsvrc_2012"; BATCH_SIZE=64; LR="3e-2"; MAX_ITER=480000; ANNEAL_FREQ=48000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Omniglot
NAME="omniglot-net"; TRAINSET="omniglot"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=3000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Aircraft
NAME="aircraft-net"; TRAINSET="aircraft"; BATCH_SIZE=8; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=3000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Birds
NAME="birds-net"; TRAINSET="cu_birds"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=3000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Textures
NAME="textures-net"; TRAINSET="dtd"; BATCH_SIZE=32; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=1500
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Quick Draw
NAME="quickdraw-net"; TRAINSET="quickdraw"; BATCH_SIZE=64; LR="1e-2"; MAX_ITER=480000; ANNEAL_FREQ=48000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Fungi
NAME="fungi-net"; TRAINSET="fungi"; BATCH_SIZE=32; LR="3e-2"; MAX_ITER=480000; ANNEAL_FREQ=15000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# VGG Flower
NAME="vgg_flower-net"; TRAINSET="vgg_flower"; BATCH_SIZE=8; LR="3e-2"; MAX_ITER=50000; ANNEAL_FREQ=1500
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

echo "All domain-specific networks are trained!"
