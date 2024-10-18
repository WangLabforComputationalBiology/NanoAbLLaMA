<h2 align="center"> NanoAbLLaMA：A Protein Large Language Model for Generate Nanobody Sequences</h2>
<h5 align="center">
  
[![Model](https://img.shields.io/badge/🤗-Model_Download-blue.svg)](https://huggingface.co/Lab608/NanoAbLLaMA)

</h5>

# Model Description
The NanoAbLLaMA is based on the ProLLaMA_stage1 and has been trained on 120K nanobody sequences for full-length nanobody sequence generation.
NanoAbLLaMA can generate sequences conditioned on germline (IGHV3-3\*01 or IGHV3S53\*01).
# Quick Inference
  ## Generate
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
  ### 3.Usage
  You can quickly start using it by simply changing the file address to the address of the model you have downloaded, and then follow the input format for input.
  If you want to perform more complex inputs, such as input files, please find the main.py file in the scripts folder to use.
  * Python
    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    # You can replace the file_path with your model path
    tokenizer = AutoTokenizer.from_pretrained("NanoAbLLaMAmodel", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("NanoAbLLaMAmodel", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    generation_config = GenerationConfig(temperature=0.2, top_k=40, top_p=0.9, do_sample=True, num_beams=1, repetition_penalty=1.2, max_new_tokens=400)

    if __name__ == '__main__':
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            raise ValueError("No GPU available.")
        model.eval()
        print("####Enter 'exit' to exit.")
        with torch.no_grad():
            while True:
                input_text = str(input("Input:"))
                if input_text.strip()=="exit":
                    break
                elif len(input_text.strip())==0:
                    break
                input_text = tokenizer(input_text, return_tensors="pt").to(device)
                generation_output = model.generate(input_text.input_ids, generation_config).to(device)
                output = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print("Output:", output)
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
      example code:train.py.
# Contact
For any questions or inquiries, please contact Haotian Chen (2394658640@qq.com) and wangxin@sztu.edu.cn
