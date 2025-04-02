from multiprocessing import Pool
from tqdm import tqdm
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from datasets import load_dataset
from openai import OpenAI
from anthropic import Anthropic
import re
import sys
import unicodedata
import os
import random
import json
import requests
import multiprocessing
import time
load_dotenv()

def return_parted_rows(df, part_ind, part_ind_list):
    df['chapter_num'] = df['current_chapter'].apply(lambda x: int(re.search(r'\d+', x).group()))
    start = part_ind_list[part_ind]
    end = part_ind_list[part_ind + 1] - 1 if part_ind + 1 in part_ind_list else df['chapter_num'].max()

    df_filtered = df[(df['chapter_num'] >= start) & (df['chapter_num'] <= end)]
    
    df_filtered = df_filtered.drop(columns=['chapter_num'])
    
    return df_filtered

def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x,y in a:
        final_answers.append(f'[{x}] : {y}')
    return "\n".join(final_answers)

    
def format_examples(examples):
    formatted_examples = []
    for row in examples:
        example = f'## Question {row["question"]} \n ## Answer {row["answer"]}'
        formatted_examples.append(example)
    return "\n".join(formatted_examples)

def extract_samples(task, numShot, model_prompt):
    questions, answer_choices, correct_answers = task_load(task, 'train')
    example_indexes = random.sample(range(len(questions)), numShot)
    example_list = []
    for i in example_indexes:
        example_list.append(model_prompt.format(question=questions[i], choices=format_choices(answer_choices[i]), answer=correct_answers[i]))
    return example_list


def task_load(task, split, new_sentences = [], engine = "", template = ""):
    if task=="medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        questions = [ds[i]['question'] for i in range(len(ds))]
        answer_choices = [ds[i]['options'] for i in range(len(ds))]
        correct_answers = [ds[i]['answer_idx'] for i in range(len(ds))]
        return questions, answer_choices, correct_answers
    
    elif task in ["medqaNoOpAlphaGen", "medqaNoOpBetaGen"]:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        questions = [ds[i]['question'] for i in range(len(ds))]
        answer_choices = [ds[i]['options'] for i in range(len(ds))]
        correct_answers = [ds[i]['answer_idx'] for i in range(len(ds))]

        for i in range(len(questions)):
            if type(new_sentences) is list:
                new_sentence = new_sentences[i]
            else:
                new_sentence = new_sentences
            sentences = re.split('(?<=[.!?]) +', questions[i])
            sentences.insert(-1, new_sentence )
            modified_paragraph = ' '.join(sentences)
            questions[i] = modified_paragraph

        return questions, answer_choices, correct_answers
    
    else:
        raise Exception("TASK NOT FOUND")

def extract_middle_paragraph(paragraph):
    paragraph += " block"
    pattern = r"^[A-Z][^A-Z]"
    if re.match(pattern, paragraph):
        paragraph = "block. " + paragraph
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    middle_part = " ".join(sentences[1:-1])

    return middle_part


def load_checkpoint(task, output_dir):
    return

def save_checkpoint(task, output_dir):
    return
    
def filterContext(context):
    end_tag = "</end>"
    if end_tag in context:
        return context.split(end_tag)[0] + end_tag
    return context

def run_inference(content, engine, temp=0.000001, max_tokens_output=1024, tokenizer=None, model=None, local=False, vllm = False):
    if local:
        messages = [{"role": "user", "content": f"{content}"}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda:0')
        outputs = model.generate(inputs, max_new_tokens=max_tokens_output, do_sample = True, temperature=temp)
        text = tokenizer.batch_decode(outputs)[0]
        return text.split("<|assistant|>")[-1]
    
    if vllm:
        return None
    
    elif "claude" in engine:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        messages = [{"role": "user", "content": f"{content}"}]
        response = client.messages.create(
            model=engine,
            max_tokens=max_tokens_output,
            temperature=temp,
            messages=messages
        )
        response_text = response.content[0].text
        return response_text
    
    elif  "o1" in engine:
        model_name = "o1-mini-2024-09-12"
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages = [{"role": "user", "content": f"{content}"}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        response_text = response.choices[0].message.content
        return response_text
    
    elif 'o3' in engine:
        model_name = engine
        if "low" in engine:
            reasoning_effort = "low"
            model_name = "o3-mini"
        elif "high" in engine:
            reasoning_effort = "high"
            model_name = "o3-mini"
        else:
            reasoning_effort = "medium"
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages = [{"role": "user", "content": f"{content}"}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            reasoning_effort=reasoning_effort
        )
        response_text = response.choices[0].message.content
        return response_text
    
    elif "deepseek" in engine:
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        messages = [{"role": "user", "content": content}]
        
        max_retries = 3   # Adjust the number of retries as needed
        delay = 5         # Delay in seconds between retries

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    stream=False
                )
                
                # Log the raw response content for debugging.
                raw_content = response.content.decode("utf-8") if response.content else ""
                print("Raw response content:", repr(raw_content))
                
                if not raw_content.strip():
                    raise Exception("Empty raw content from API")
                
                # Parse the response JSON manually
                parsed = response.json()
                
                # Check if the expected 'choices' key exists and contains data
                if "choices" not in parsed or not parsed["choices"]:
                    raise Exception("Response missing 'choices'")
                
                # Extract the response text
                response_text = parsed["choices"][0]["message"]["content"]
                if not response_text:
                    raise Exception("Empty response text in API reply")
                
                return response_text

            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise Exception("All retries failed; API did not return a valid response.")

    elif "gpt" in engine:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages = [{"role": "user", "content": f"{content}"}]
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens_output,
            frequency_penalty=0.0
        )
        response_text = response.choices[0].message.content
        return response_text
    else:
        print("ENGINE not found.")
    
def run_inference_worker(args):
    i, j, model_prompt, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model = args
    text = run_inference(model_prompt, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model)
    query_object = {'id': ('attempt_' + str(j)), 'COT': text}
    return i, query_object

def parallelize_inference(model_db, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, NUMBER_OF_ENSEMBLE):
    D = {'attempts': []}
    tasks = []
    for i in range(len(model_db)):
        model_prompt = model_db[i]['model_prompt']
        for j in range(NUMBER_OF_ENSEMBLE):
            tasks.append((i, j, model_prompt, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model))

    if "gpt-4o" in ENGINE and "mini" not in ENGINE:
        cpu_count = min(5, multiprocessing.cpu_count())
    elif "claude" in ENGINE or "o3" in ENGINE or "o1" in ENGINE or "4o" in ENGINE:
        cpu_count = min(5, multiprocessing.cpu_count())
    else:
        cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} workers to make this shi faster")
    with Pool(processes=cpu_count) as pool:
        results = []
        for result in tqdm(pool.imap(run_inference_worker, tasks), total=len(tasks)):
            results.append(result)

    for i, query_object in results:
        model_db[i]['attempts'].append(query_object)

    return model_db

def run_vllm(model_db, NUMBER_OF_ENSEMBLE, llm, sampling_params):
    prompts = []
    for i in range(len(model_db)):
        model_prompt = model_db[i]['model_prompt']
        prompts.append(model_prompt)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    print("Generation Complete")

    for i in range(len(model_db)):
        for j in range(NUMBER_OF_ENSEMBLE):
            output = outputs[i*NUMBER_OF_ENSEMBLE+j]
            generated_text = output.outputs[0].text
            query_object = {'id': ('attempt_' + str(j)), 'COT': generated_text}
            model_db[i]['attempts'].append(query_object)

    print("Model_db generation assignmented complete.")
    return model_db

def get_text_by_customid(records, target_customid):
    for record in records:
        if record.get("custom_id") == target_customid:
            return record["result"]["message"]["content"][0].get("text")
    return None

def pull_from_batch(model_db, model_name, task):
    output_dir = "RAG_Pull/batch_outputs/"
    file_name = output_dir + "batch_" + model_name
    if "alphagen2" in task.lower():
        file_name += "_AlphaGen2.jsonl"
    else:
        file_name += "_BetaGen2.jsonl"
    with open(file_name, 'r', encoding='utf-8') as file:
        records = [json.loads(line) for line in file if line.strip()]

    for i in range(len(model_db)):
        for j in range(1):
            generated_text = get_text_by_customid(records,str(i+1))
            query_object = {'id': ('attempt_' + str(j)), 'COT': generated_text}
            model_db[i]['attempts'].append(query_object)
    return model_db
    
class MultiChoiceFilter:
    def __init__(self, ignore_case=False, ignore_punctuation=False, regex_pattern=r"[\(\[]([A-Z])[\)\]]"):
        
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                                       if unicodedata.category(chr(i)).startswith("P"))

    def filter_text(self, text):
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            text = text.translate(self.punct_tbl)
        return text

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match

    def extract_answer(self, response, choices=None, final_letter = 'D'):
        matchFirst = re.search(r'the answer is ([A-D])', response)
        if matchFirst:
            return f"({matchFirst.group(1)})"
        
        matchFifth = re.search(r'answer is ([A-D])\.?', response, re.IGNORECASE)
        if matchFifth:
            return f"({matchFifth.group(1)})"
        
        matchSecond = re.search(r'Therefore, the final model answer is ([A-D])',response, re.IGNORECASE)
        if matchSecond:
            return f"({matchSecond.group(1)})"
        
        matchSeventh = re.search(r'boxed\{([A-D])\}', response)
        if matchSeventh:
            return f"({matchSeventh.group(1)})"
        
        match12 = re.search(r'The answer is \*\*\[([A-D])\]', response)
        if match12:
            return f"({match12.group(1)})"
        
        matchNine = re.search(r'Answer: ([A-D])', response, re.IGNORECASE)
        if matchNine:
            return f"({matchNine.group(1)})"
        
        matchTen = re.search(r'Answer:\*\* ([A-D])', response, re.IGNORECASE)
        if matchTen:
            return f"({matchTen.group(1)})"

        matchEleven = re.search(r'is ([A-D])', response)
        if matchEleven:
            return f"({matchEleven.group(1)})"

        matchesThird = self.regex.findall(response) 
        if matchesThird:
            return f"({matchesThird[-1]})"
        
        matchFourth = re.search(r'\*\*([A-D])\*\*', response)
        if matchFourth:
            return f"({matchFourth.group(1)})"
        
        matchSixth = re.search(r'\[([A-D])\]', response)
        if matchSixth:
            return f"({matchSixth.group(1)})"

        matchEight = re.search(r'Choice ([A-D])', response, re.IGNORECASE)
        if matchEight:
            return f"({matchEight.group(1)})"

        match13 = re.search(r'\[\*\*([A-D])\]', response, re.IGNORECASE)
        if match13:
            return f"({match13.group(1)})"

        match14= re.search(r'\[([A-D])\*\*\]', response, re.IGNORECASE)
        if match14:
            return f"({match14.group(1)})"
        
        match15= re.search(r'My Answer\n\*\*([A-D])', response, re.IGNORECASE)
        if match15:
            return f"({match15.group(1)})"
        
        #\n\n**Answer:**\nA
        match16= re.search(r'\n\n\*\*Answer:\*\*\n([A-D])', response, re.IGNORECASE)
        if match16:
            return f"({match16.group(1)})"
        
        return "[invalid]"

    def filter_responses(self, responses, choices):
        return [self.extract_answer(resp, choices) for resp in responses]
        
def select_after_backslash(s):
    parts = s.split('/', 1) 
    if len(parts) > 1:
        return parts[1]
    else:
        return s
    
def contains_any(strings_list, singular_string):
    return any(substring in singular_string for substring in strings_list)
