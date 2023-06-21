# jsonformer-sample

To run the `jsonformer_sample.py` script, open your terminal and execute the following command:

```bash
python jsonformer_sample.py
```

This will produce the following output:

```
===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
binary_path: C:\Users\ernes\scoop\apps\python\current\Lib\site-packages\bitsandbytes\cuda_setup\libbitsandbytes_cuda116.dll
CUDA SETUP: Loading binary C:\Users\ernes\scoop\apps\python\current\Lib\site-packages\bitsandbytes\cuda_setup\libbitsandbytes_cuda116.dll...
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:21<00:00,  5.27s/it]
WARNING:root:For training, the BetterTransformer implementation for gpt_neox  architecture currently does not support padding as fused kernels do not support custom attention masks. Beware that passing padded batched training data may result in unexpected outputs.
{'agent': 'Dolly', 'personality': 'cheerful', 'defaultMessage': 'Hello, World!'}
```

## Screenshot

![./Screenshot 2023-06-20 204612.png](Screenshot%202023-06-20%20204612.png)
