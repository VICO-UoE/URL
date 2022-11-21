################### Test URL Model with Task-specific Adapters (TSA) ###################
# specify a pretrained model by --model.name and --model.dir
# choose single/multi-domain setting by --test.mode: sdl or mdl
# specify feature or classifier adaptation by --test-opt: alpha, beta, or alpha+beta
# adapter connection: serial, or residual
# adapter parameterization: matirx, or vector (channel-wise)
# adapter initialization: eye (identity), or random

# an example for the best configuration
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
CUDA_VISIBLE_DEVICES=<gpu-id> python test_extractor_tsa.py --model.name=url --model.dir ./saved_results/url \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode mdl