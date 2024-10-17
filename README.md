<h2 align="center"> NanoAbLLaMA：A Protein Large Language Model for Generate Nanobody Sequences</h2>
<h5 align="center">
  
[![Model](https://img.shields.io/badge/🤗-Model_Download-blue.svg)](https://huggingface.co/Lab608/NanoAbLLaMA)

</h5>

# Model Description
The NanoAbLLaMA is based on the ProLLaMA_stage1 and has been trained on 120K nanobody sequences for full-length nanobody sequence generation.
NanoAbLLaMA can generate sequences conditioned on germline (IGHV3-3*01 or IGHV3S53*01).
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
  * Python
    ```python
    import argparse
    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from transformers import GenerationConfig
    
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=400
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="NanoAbLLaMAmodel", type=str, help="The local path of the model.")
    args = parser.parse_args()
    
    load_type = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            quantization_config=None,
            device_map="auto"
        )
    
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    
    if __name__ == '__main__':
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            raise ValueError("No GPU available.")
    
        model.eval()
        with torch.no_grad():
            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip())==0:
                    break
                input_text = raw_input_text
                input_text = tokenizer(input_text,return_tensors="pt").to(device)
                generation_output = model.generate(
                            input_ids = input_text["input_ids"].to(device),
                            attention_mask = input_text['attention_mask'].to(device),
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            generation_config = generation_config,
                            output_attentions = False
                        )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                print(output)
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
