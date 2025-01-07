from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

import pandas as pd
from tqdm import tqdm
import csv
import re

#### Data

data = pd.read_csv("/home/chayan/PromptEng25 Workshop/Data/Pos_PTX_Prompt_Test.csv")

Original_report = data["ReportText"].reset_index(drop=True)   ### 

Acc = data["AccessionNumber"].reset_index(drop=True) ## for testing [0:10] 

print(len(Original_report))

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Open CSV file for writing

csv_file_path = "/home/chayan/PromptEng25 Workshop/Data/Pos_PTX_Prompt_Test_Findings_Impression.csv"


with open(csv_file_path, mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(['Acc_Number', 'Original_Report', 'Generated_Report', 'Impression'])


for j in tqdm(range(len(Original_report))):
    messages_first = [
        {"role": "system", "content": """You are analyzing a radiology report for a chest X-ray examination. Extract Clinical Notes, and  Findings Sections from the full report. Anonymize the report excluding Dr or Patients Name. 
                    Categorize your output into following two sections only avoiding any note or extra writings or repeating same sentence in both categories:
    1. Clinical Notes: Identify the clinical notes or history or indication only. If no notes are given just response as "Not mentioned".  
    2. Findings: Identify main findings of the report including impression or conclusion.
     """},
        {"role": "user", "content": f"Extract the clinical notes, and findings of this report must excluding Dr name. {Original_report[j]}." 
    },
    ]

    input_ids_first = tokenizer.apply_chat_template(
        messages_first,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs_first = model.generate(
        input_ids_first,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    output = outputs_first[0][input_ids_first.shape[-1]:]
    response_first = tokenizer.decode(output, skip_special_tokens=True)
    
    # Second LLM message: Generate final findings without headings
    messages_second = [
        {"role": "system", "content": """You are reading the extracted Clinical notes and Findings of a radiology report. Your task is to identify **all abnormalities** (including mild or subtle findings such as mild enlargement or changes in size, shape, or structure, post operative status.) and **support and monitoring devices** (e.g., catheters, wires, pacemakers, NG tubes).
                                         Categorize your output into the impresion sections only avoiding any note or extra writings:
                                         **Impression**: Include all abnormal findings and support devices that suggest any deviation from normal, even if described as mild or subtle. If no abnormalities present, just write 'No abnormal findings'.
                                         """},                                                                   
        {"role": "user", "content": f"Find the abnormalities of this report. {response_first}"}
    ]

    input_ids_second = tokenizer.apply_chat_template(
        messages_second,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs_second = model.generate(
        input_ids_second,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.07,
        top_p=0.9,
    )
    output_second = outputs_second[0][input_ids_second.shape[-1]:]
    final_response = tokenizer.decode(output_second, skip_special_tokens=True)
    

    # Compile results and write to CSV
    results = [Acc[j], Original_report[j], response_first, final_response]
    with open(csv_file_path, mode='a', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(results)