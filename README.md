# Binarized Neural Networks Architecutre Search for Semantic Segmentation  
 * Configure your network architecture in the bash file and then run it
    * Gradient based search
      ```
      CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_darts_search.bash 
      ```
    * Performance based search
      ```
      CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_bnas_search.bash 
      ```
* Evaluate your search outcome 
  * Configure your evaluation network in run_darts_eval.bash and then run it
    ```
    CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_darts_eval.bash 
    ```
# Acknowledgement
  * Differentiable architecture search for convolutional and recurrent networks. ([DARTS](https://github.com/quark0/darts))
  * This repository contains an unofficial implementation of [BNAS](https://arxiv.org/abs/1911.10862)
  * Binarized Encoder-Decoder Network and Binarized Deconvolution Engine for Semantic Segmentation. ([BEDN](https://github.com/penpaperkeycode/BEDN)) 
  * Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation. ([SBNN](https://bitbucket.org/jingruixiaozhuang/group-net-semantic-segmentation/src/master/))
  * [Flops Counter](https://github.com/sovrasov/flops-counter.pytorch)
  * [Segmenation metrics](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py)
  * [ENet](https://github.com/osmr/imgclsmob)
  * [DAPNet](https://github.com/Reagan1311/DABNet)