<h2 align="center"> NanoAbLLaMA：A Protein Large Language Model for Generate Nanobody Sequences</h2>
<h5 align="center">
  
[![Model](https://img.shields.io/badge/🤗-Model_Download-blue.svg)](https://huggingface.co/Lab608/NanoAbLLaMA)

</h5>

# Model Description
The NanoAbLLaMA is based on the ProLLaMA_stage1 and has been trained on 120K nanobody sequences for full-length nanobody sequence generation.
NanoAbLLaMA can generate sequences conditioned on germline (IGHV3-3\*01 or IGHV3S53\*01).
# Quick Inference
  ## Usage
  ### 1.Install Requirements
  * torch==2.0.1
  * transformers==4.31.0
  * cuda==11.7
  ```bash
  git clone https://github.com/WangLabforComputationalBiology/NanoAbLLaMA.git
  cd NanoAbLLaMA
  pip install -r requirements.txt
  ```
  ### 2.Download Model
  Download from [Hugging Face](https://huggingface.co/Lab608/NanoAbLLaMA)
  ### 3.Generate
  1. Replace the model path with your local path.
  2. Run `./scripts/main.py` and follow the input format for input.
  * Example:
  ### 4.Input Format
  The instructions which you input to the model should follow the following format:
  ```text
  [Generate by germline] Germline=<IGHV3-3*01>
  or
  [Generate by germline] Germline=<IGHV3S53*01>
  ```
  ```text
  #You can also specify the first few amino acids of the protein sequence:
  [Generate by germline] Germline=<IGHV3-3*01> Seq=<QVQL
  ```
  ## Training
  1. Process the train_dataset into a format similar to our example.json format and put the train_datasets under `./data/instruction_tuning_dataset`. We provided `./data/example.json` as an example.
  2. Run `./scripts/train.py` and specify the paths for input_file and output_file.
# Contact
For any questions or inquiries, please contact Haotian Chen (2394658640@qq.com) and wangxin@sztu.edu.cn
