# [NeurIPS 2024] How do Large Language Models Handle Multilingualism?

This repository contains code for the paper "[How do Large Language Models Handle Multilingualism?](https://arxiv.org/abs/2402.18815)". Below is our hypothesized multilingual workflow MWork.

<img src="./figures/mwork.png" alt="./" style="zoom:63%;" />

## Installation

The package can be installed by running the following command at the root of this repository: 

```shell
conda create -n SeaExam python=3.9
conda activate SeaExam
pip install -r requirement.txt
```

## Layer Embedding Decoding

Figure 1 in the paper is obtained by decoding the embedding by decoding the hidden embeddings of each layer to tokens within the LLM's vocabulary.

Use the following command to run the experiment:

```sh
cd layers
python test_layer.py
```

For possible command parameters, run

```sh
python test_layer.py --help
```

<img src="./figures/layer.png" alt="./" style="zoom:80%;" />

## Neuron Detection (PLND) 

Neuron detection for MLP components is supported for arbitrary models. For attention components, neuron detection is supported for Llama, Mistral and Gemma.

The corpus for neuron detection is stored in `./neuron_detection/corpus_all`.

```sh
cd neuron_detection
python neuron_detection.py
```

For possible command parameters and components to detect neurons in, run

```sh
python neuron_detection.py --help
```

The detected neurons will be stored in folder `./output_neurons`.

## Neuron Deactivation

We provide codes for detecting neurons in Llama, Mistral and Gemma.

### Running

We need to  **change transformers package**. 

```sh
cd /neuron_deactivate
python test_mistral_gsm.py {language} {understanding layer} {generation layer} {attn deact_number} {ffn deact_number} {whether under_attn} {whether reason_attn} {whether gen_attn} {whether under_ffn} {whether reason_ffn} {whether gen_ffn}
```

## Neuron Specific Enhancement

Neuron specific tuning code is the same for all models.

### Running

We need to  **change transformers package**. 

```sh
cd /neuron_enhancement
python train_neuron.py
```

### Parameters

Note that `attn_k` and `attn_v` needs to be  divided by `kv_repeat`. `index_keys` requires fitting to model you want to train and number of understanding layer and generation layer needs to be changed correspondingly.

```python
index_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]         

index_keys_under = [i for i in range(8)]
index_keys_gen = [31-i for i in range(4)]

attn_k = {key: {num//4 for num in value} for key, value in attn_k.items()}
attn_v = {key: {num//4 for num in value} for key, value in attn_v.items()}
```

## Citation

If you found this repository useful, please consider

```latex
@inproceedings{zhao2024large,
  title={How do Large Language Models Handle Multilingualism?},
  author={Zhao, Yiran and Zhang, Wenxuan and Chen, Guizhen and Kawaguchi, Kenji and Bing, Lidong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
