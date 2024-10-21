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
  1. You must change the model path to your local path, with the default being `./NanoAbLLaMAmodel`.
  2. Run `./scripts/main.py` and follow the input format for input.
  * Options:
  ```text
  usage: main.py [-h] [--model] [--temperature] [--top_k] [--top_p] [--do_sample] [--repetition_penalty] [--interactive] [--input_file] [--output_file]

  options:
  -h, --help show this help message and exit
  --model The local path of the model. (default: NanoAbLLaMAmodel)
  --temperature The value used to regulate the probability of the next token; a higher temperature leads to more diverse text, but it may also result in untrustworthy content. (default: 0.2)
  --top_k The number of top-probability word tokens to retain for top-k filtering. (default: 40)
  --top_p If set to a floating-point number less than 1, only the most probable tokens whose cumulative probability reaches top_p or higher are retained for generation. (default: 0.9)
  --do_sample Whether to use sampling; otherwise, use greedy decoding. (default: True)
  --repetition_penalty The parameter for repetition penalty, 1.0 indicates no penalty. (default: 1.2)
  --interactive If True, you can input instructions interactively. If False, the input instructions should be in the input_file. (default: True)
  --input_file You can put all your input instructions in this file (one instruction per line). (default: None)
  --output_file All the outputs will be saved in this file. (default: None)
  ```
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
