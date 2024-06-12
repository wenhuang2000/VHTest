# Visual Hallucinations of Multi-modal Large Language Models (2024 ACL Findings)

![image](Illustrations/4_examples.png)

## Introduction

Visual hallucination (VH) means that a multi-modal LLM (MLLM) imagines incorrect details about an image in visual question answering. Existing studies find VH instances only in existing image datasets, which results in biased understanding of MLLMs' performance under VH due to limited diversity of such VH instances. In this work, we propose a tool called VHTest to generate a diverse set of VH instances. Specifically, VHTest finds some initial VH instances in existing image datasets (e.g., COCO), generates a text description for each VH mode, and uses a text-to-image generative model (e.g., DALL-E-3) to generate VH images based on the text descriptions. We collect a benchmark dataset with 1,200 VH instances in 8 VH modes using VHTest. We find that existing MLLMs such as GPT-4V, LLaVA-1.5, and MiniGPT-v2 hallucinate for a large fraction of the instances in our benchmark. Moreover, we find that fine-tuning an MLLM using our benchmark dataset reduces its likelihood to hallucinate without sacrificing its performance on other benchmarks. Our benchmarks are provided in this repo.

## Benchmarks

### Overview

VHTest offers two benchmarks:
- Open-Ended Question (OEQ) Benchmark
- Yes/No Question (YNQ) Benchmark
Details can be found in the 'benchmarks' folder.


### Beginning
1. Clone this repository and navigate to VHTest folder
```
git clone https://github.com/wenhuang2000/VHTest.git
cd VHTest
```
2. Create an environment for VHTest
```
conda create -n vhtest python=3.10 -y
conda activate vhtest
pip install --upgrade pip
```

### Evaluate LLaVA
1. Install packages of LLaVA
```
cd LLaVA
pip install -e .
pip install flash-attn --no-build-isolation
cd ..
```

2. Evaluate LLaVA on OEQ version Benchmark
To evaluate LLaVA on the OEQ version benchmark of VHTest, run
```
python -m evaluation.LLaVA.llava_oeq_version --model-path PATH_TO_MODEL_EVALUATED --image-path PATH_TO_VHTEST_IMAGES --test-xlsx-path PATH_TO_VHTEST_QUESTION_TABLES --results-xlsx-path PATH_TO_SAVE_RESULTS
```
For example,
```
python -m evaluation.LLaVA.llava_oeq_version --model-path "liuhaotian/llava-v1.5-13b" --image-path ./Benchmark/Images --test-xlsx-path ./Benchmark/OEQ_Benchmark --results-xlsx-path ./Results/llava-v1.5-13b/OEQ_Benchmark
```

3. Evaluate LLaVA on YNQ version Benchmark
To evaluate LLaVA on the YNQ version benchmark of VHTest, run
```
python -m evaluation.LLaVA.llava_ynq_version --model-path PATH_TO_MODEL_EVALUATED --image-path PATH_TO_VHTEST_IMAGES --test-xlsx-path PATH_TO_VHTEST_QUESTION_TABLES --results-xlsx-path PATH_TO_SAVE_RESULTS
```
For example,
```
python -m evaluation.LLaVA.llava_ynq_version --model-path "liuhaotian/llava-v1.5-13b" --image-path ./Benchmark/Images --test-xlsx-path ./Benchmark/YNQ_Benchmark --results-xlsx-path ./Results/llava-v1.5-13b/YNQ_Benchmark
```

## License

Distributed under MIT License. See `LICENSE` for more information.

## Citation

If you find VHTest is helpful, please cite:

```bibtex
@article{huang2024visual,
  title={Visual Hallucinations of Multi-modal Large Language Models},
  author={Huang, Wen and Liu, Hongbin and Guo, Minxin and Gong, Neil Zhenqiang},
  journal={arXiv preprint arXiv:2402.14683},
  year={2024}
}
```
