################### Test URL Model with Task-specific Adapters (TSA) ###################
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
CUDA_VISIBLE_DEVICES=<gpu-id> python test_extractor_tsa.py --model.name=url --model.dir ./saved_results/url \
-test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye