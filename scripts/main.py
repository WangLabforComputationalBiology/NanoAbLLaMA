import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="model", type=str, help="The local path of the model.")
parser.add_argument('--temperature', default=0.2, type=float, help="The value used to regulate the probability of the next token; a higher temperature leads to more diverse text, but it may also result in untrustworthy content.")
parser.add_argument('--top_k', default=40, type=int, help="The number of top-probability word tokens to retain for top-k filtering.")
parser.add_argument('--top_p', default=0.9, type=float, help="If set to a floating-point number less than 1, only the most probable tokens whose cumulative probability reaches top_p or higher are retained for generation.")
parser.add_argument('--do_sample', default=True, action='store_true', help="Whether to use sampling; otherwise, use greedy decoding.")
parser.add_argument('--repetition_penalty', default=1.2, type=float, help="The parameter for repetition penalty, 1.0 indicates no penalty.")
parser.add_argument('--interactive', default=True, action='store_true', help="If True, you can input instructions interactively. If False, the input instructions should be in the input_file.")
parser.add_argument('--input_file', default=None, help="You can put all your input instructions in this file (one instruction per line).")
parser.add_argument('--output_file', default=None, help="All the outputs will be saved in this file.")
args = parser.parse_args()

generation_config = GenerationConfig(
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    do_sample=args.do_sample,
    repetition_penalty=args.repetition_penalty,
    max_new_tokens=400
)

model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=None,
        device_map="auto"
    )

tokenizer = LlamaTokenizer.from_pretrained(args.model)

if __name__ == '__main__':
    if args.interactive and args.input_file:
        raise ValueError("interactive is True, but input_file is not None.")
    if (not args.interactive) and (args.input_file is None):
        raise ValueError("interactive is False, but input_file is None.")
    if args.input_file and (args.output_file is None):
        raise ValueError("input_file is not None, but output_file is None.")

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        raise ValueError("No GPU available.")

    if args.interactive:
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
    else:
        outputs=[]
        with open(args.input_file, 'r') as f:
            examples =f.read().splitlines()
        print("Start generating...")
        for index, example in tqdm(enumerate(examples),total=len(examples)):
            input_text = tokenizer(example,return_tensors="pt")

            generation_output = model.generate(
                input_ids = input_text["input_ids"].to(device),
                attention_mask = input_text['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            outputs.append(output)
        with open(args.output_file,'w') as f:
            f.write("\n".join(outputs))
        print("All the outputs have been saved in",args.output_file)
