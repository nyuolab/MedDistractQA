from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from RAG_Pull.prompts import *
from RAG_Pull.utils import *
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
import gc
import json
import pandas as pd
import pprint as p
import time
import random
import argparse
from typing import List, Dict, Any
import os
load_dotenv()
import lipsum




def main(ENGINE, TASK_LIST, SENTENCE_LIST, RAG=False, RAG_NUM_RANGE = [0], EMBED_RAG = False, USE_RAG_SUM = False, EMBED_SENTENCE = True, dir_app = "", PROMPT_TO_USE = prompt_eval_mod):
    large_model_vllm = False
    if contains_any(["claude"],  ENGINE.lower()):
        INF_METHOD = "batch_server"
        print("PULLING INFERENCE FROM BATCHES")
    elif contains_any(["gpt", "o3-mini", "o1-mini","llama3-3-70b-chat", "llama3-70b-chat", "claude","llama3-3-70B-DSR1", "llama3-3", "deepseek-chat"],  ENGINE.lower()):
        INF_METHOD = 'server'
        print("RUNNING INFERENCE NODE IMPLEMENTATION")
    elif "70b" in ENGINE.lower():
        large_model_vllm = True
        INF_METHOD = 'vllm'
        print("RUNNING VLLM IMPLEMENTATION")
    elif contains_any(["meerkat","llama-3.2", "phi", "qwen", "gemma", "mistral", "llama-2", "meta-llama-3-8b", "ultramedical", "llama-3.1-8b", "phi-3-small", "phi-3-medium", "deepseek-r1-distill"], ENGINE.lower()):
        INF_METHOD = 'vllm'
        print("RUNNING VLLM IMPLEMENTATION")
    elif contains_any(["medmobile"], ENGINE.lower()):
        INF_METHOD = 'local'
        print("RUNNING LOCAL IMPLEMENTATION")
    else:
        INF_METHOD = 'invalid'
        raise Exception(f"{ENGINE.lower()} is not a valid inference method.")
    SPLIT = "test"
    NUMBER_OF_ENSEMBLE = 1
    ENGINE_TEMPERATURE = 0.000000001 
    MAX_TOKEN_OUTPUT = 8000
    NSHOT = 0
    STOP_GEN = 5 ## For testing purposes; stop generating after {STOP_GEN} amount of test-questions
    OUTPUT_DIR =  ## SET OUTPUT DIR
    CONTEXT_DIR = ## SET RAG RETRIEVAL DIR

    ## APPLY RAG
    if RAG:
        OUTPUT_DIR += "RAG_"
        MAX_NUMBER_OF_CONTEXT_PARA = 1
        SEARCH_ALGO = "RAG" ## Change to "None" if no context,["BM25", "RAG", "RAG_Title_LLM", "RAGBM25_take_top_5_each", "RAGBM25_lowest_index_sum", "RAGBM25_RRF"]
    else:
        OUTPUT_DIR += dir_app
        SEARCH_ALGO = "None" ## Change to "None" if no context,["BM25", "RAG", "RAG_Title_LLM", "RAGBM25_take_top_5_each", "RAGBM25_lowest_index_sum", "RAGBM25_RRF"]

    ## INIATIATE OUTPUT DB
    results_db = {
        "metadata": {
            "model" : select_after_backslash(ENGINE),
            "temperature" : ENGINE_TEMPERATURE,
            "num_shot" : NSHOT,
            "number_of_ensemble": NUMBER_OF_ENSEMBLE,
            "max_tokens" : MAX_TOKEN_OUTPUT,
        }
    }

    ## SET FILE DIRECTORY PATHS (Context dir, file output)
    if SEARCH_ALGO != "None":

        runName = f'({ENGINE}) COT simple prompt'
        if NUMBER_OF_ENSEMBLE >1:
            runName += f" + Ensemble ({NUMBER_OF_ENSEMBLE})"
            
        if RAG:
            runName += " + RAG"

        contextdf_path = f'{CONTEXT_DIR}{SPLIT}_{SEARCH_ALGO}_10000.csv'
        contextdf = pd.read_csv(contextdf_path)
        print("RAG df IMPORTED.")
        results_db['metadata']['context_search_algo'] = SEARCH_ALGO
        results_db['metadata']['path_of_context'] = contextdf_path
        results_db['metadata']['number_of_context_paras'] = MAX_NUMBER_OF_CONTEXT_PARA

    else:   
        if NUMBER_OF_ENSEMBLE > 1:
            runName = f' ({ENGINE}) + Ensemble ({NUMBER_OF_ENSEMBLE})'
        else: 
            runName = f' ({ENGINE})'

    ## DISPLAY HYPERPARAMETERS
    for name, value in results_db['metadata'].items():
        print(f"{name} : {value}")

    ## LOAD IN MODEL IF VLLM/LOCAL
    if INF_METHOD == 'vllm':
        model_path = ENGINE
        if contains_any(["qwen"], ENGINE.lower()):
            sampling_params = SamplingParams(temperature=ENGINE_TEMPERATURE, top_p=1,repetition_penalty=1.05, max_tokens = MAX_TOKEN_OUTPUT)
            print("Adjusting sampling params to include repitition penalty for qwen")
        else:
            sampling_params = SamplingParams(temperature=ENGINE_TEMPERATURE, top_p=1, max_tokens = MAX_TOKEN_OUTPUT)
        if large_model_vllm:
            llm = LLM(model=model_path, tensor_parallel_size=4)
        else:
            llm = LLM(model=model_path)
        print("VLLM model loaded in.")

    elif INF_METHOD == 'server' or INF_METHOD == "batch_server":
        model = None
        tokenizer = None

    else:
        model_path = ENGINE
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda",torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)


    ## OUTPUT RUN INFO:
    print("Model Running: " + ENGINE)
    print("Run: " + runName)

    ## ASSIGN EVAL FILTER
    mcf = MultiChoiceFilter(ignore_case=True, ignore_punctuation=True)
    print("Number of tasks: " + str(len(TASK_LIST)))

    for SENTENCE_RANK, sentence in enumerate(SENTENCE_LIST):
        for task in TASK_LIST:
            for RAG_RANK in RAG_NUM_RANGE:
                print(f'Sentence Rank/Rag rank: {SENTENCE_RANK}/{RAG_RANK}')
                contexts = []
                if SEARCH_ALGO != "None":
                    for i in range(len(contextdf)):
                        context = ""
                        for j in range(RAG_RANK, RAG_RANK+MAX_NUMBER_OF_CONTEXT_PARA):
                            summary = contextdf[f'para_{j}'][i]
                            context = context + " " + (summary)
                        contexts.append(extract_middle_paragraph(context))
                if RAG and EMBED_RAG:
                    question_list, answer_choices_list, correct_answer_list = task_load(task, SPLIT, contexts)
                    print("Adding RAG into questions")
                elif EMBED_SENTENCE:
                    question_list, answer_choices_list, correct_answer_list = task_load(task, SPLIT, sentence)
                else:
                    question_list, answer_choices_list, correct_answer_list = task_load(task, SPLIT, "")

                print(f"{task} loaded succesfully. Now conducting evaluation on {len(question_list)} samples.")

                ## CREATE MODEL_DB
                model_db = []

                for i, (question, answer_choices, correct_answer) in tqdm(enumerate(zip(question_list, answer_choices_list, correct_answer_list))):
                    D = {}
                    if EMBED_RAG or EMBED_SENTENCE:
                        if NSHOT == 0:
                            prompt = PROMPT_TO_USE
                        else: 
                            prompt = prompt_eval_with_examples
                    else:
                        if NSHOT == 0:
                            prompt = PROMPT_TO_USE
                        else: 
                            prompt = prompt_eval_with_context_and_examples

                    if NSHOT != 0:
                        examples = extract_samples(task, NSHOT, prompt_example)
                        model_prompt = prompt.format(
                            question=question,
                            choices=format_choices(answer_choices),
                            examples = ("\n").join(examples),
                            context = filterContext(contexts[i])
                        )
                    elif RAG and not EMBED_RAG:
                        model_prompt = prompt.format(question=question, choices=format_choices(answer_choices), context = filterContext(contexts[i]))
                    elif not EMBED_SENTENCE and not RAG:
                        model_prompt = prompt.format(question=question, choices=format_choices(answer_choices), context = filterContext(sentence))
                    else:
                        model_prompt = prompt.format(question=question, choices=format_choices(answer_choices), context = "")
                    ## Create question_dict that will eventually get added to master list of dict (model_db)
                
                    D['query'] = question
                    D['question_choices'] = answer_choices
                    D['correct_answer'] = correct_answer
                    D['attempts'] = []
                    D['model_prompt'] = model_prompt

                    if INF_METHOD == 'local':
                        for j in range(NUMBER_OF_ENSEMBLE):
                            text = run_inference(model_prompt, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, True)
                            query_object = {'id': ('attempt_'+str(j)), 'COT': text}
                            D['attempts'].append(query_object)

                    model_db.append(D)
                print("model_db initialized.")

                start_time = time.time() 
                if INF_METHOD == 'server':
                    model_db = parallelize_inference(model_db, ENGINE, ENGINE_TEMPERATURE, MAX_TOKEN_OUTPUT, tokenizer, model, NUMBER_OF_ENSEMBLE)
                elif INF_METHOD == 'vllm':
                    model_db = run_vllm(model_db, NUMBER_OF_ENSEMBLE, llm, sampling_params)
                elif INF_METHOD == "batch_server":
                    model_db = pull_from_batch(model_db, ENGINE, task)
                    ## THIS IS ASSUMING THAT A BATCH FILE ALREADY EXISTS AND IS PROCESSED. REQUIRES TWO INDIVIDUAL STEPS FOR THIS METHOD
                else:
                    print("Method of inference not supported")
                end_time = time.time()
                print(f'Inference took {end_time-start_time} seconds using {INF_METHOD}')

                total_num_ques = 0
                num_correct = 0
                num_invalid = 0
                for q in model_db: 
                    choices = q['question_choices']
                    letter_counts = {}
                    for attempt in q['attempts']:
                        attempt['model_choice'] = mcf.extract_answer(attempt['COT'], choices)
                        if attempt['model_choice'] in letter_counts:
                            letter_counts[attempt['model_choice']] += 1 
                        else:
                            letter_counts[attempt['model_choice']] = 1
                    max_count = 0
                    for letter, count in letter_counts.items():
                        if count > max_count:
                            q['ensemble_answer'] = letter
                            max_count = count
                    total_num_ques+=1
                    if q['ensemble_answer'].strip("()") == q['correct_answer']:
                        num_correct += 1
                    elif q['ensemble_answer'] == "[invalid]":
                        num_invalid += 1
                
                print("Number of correct answer: " + str(num_correct))
                print("Total number of questions: " + str(total_num_ques))
                print("Model Accuracy: " + str(num_correct/total_num_ques))
                
                results_db_task = results_db.copy()
                results_db_task['metadata']['informal_run_name'] = runName
                results_db_task['metadata']['rag_rank'] = RAG_RANK
                results_db_task['metadata']['task'] = task
                results_db_task['metadata']['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                results_db_task['metadata']['prompt'] = prompt
                results_db_task['metadata']['number_of_invalids'] = num_invalid
                results_db_task['metadata']['number_of_questions'] = total_num_ques
                results_db_task['metadata']['true_accuracy'] = num_correct/total_num_ques
                results_db_task['metadata']['eff_accuracy'] = num_correct/(total_num_ques-num_invalid)
                results_db_task['metadata']['run_time'] = end_time-start_time
                results_db_task['metadata']['run_time_per_iteration'] = (end_time-start_time)/total_num_ques
                results_db_task['metadata']['nonsense_sentence'] = sentence
                results_db_task['metadata']['inference_method'] = INF_METHOD
                results_db_task['metadata']['embed_rag'] = EMBED_RAG
                results_db_task['metadata']['sentence_rank'] = SENTENCE_RANK
                results_db_task['model_results'] = model_db

                filename = f"{OUTPUT_DIR}{task}/query_database_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as file:
                    json.dump(results_db_task, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("engine", type=str, help="Name of the engine")
    parser.add_argument(
        "eval_type", 
        type=str, 
        help="Name of the evaluation", 
        default="default_eval"
    )
    args = parser.parse_args()
    
    ## DATASET CAN BE FOUND AT HUGGINGFACE HUB AT "KrithikV/MedDistractQA"
    file_path = "DOWNLOAD MedDistractQA-Nonliteral AND PLACE PATH HERE"
    with open(file_path, 'r') as file:
        data = json.load(file)
    CONFOUNDER_SENTENCES_ALPHA = [entry["confounder_sentence"] for entry in data]
    
    file_path = "DOWNLOAD MedDistractQA-Bystander AND PLACE PATH HERE"
    with open(file_path, 'r') as file:
        data = json.load(file)
    CONFOUNDER_SENTENCES_BETA = [entry["confounder_sentence"] for entry in data]

    if args.eval_type == "baseline":
        main(args.engine, ["medqa"], [""])

    elif args.eval_type == "ragBase":
        main(args.engine, ["medqa"], [""], True, [0])

    elif args.eval_type == "rag":
        main(args.engine, ["medqa"], [""], True, [1,10,20,25,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000])

    elif args.eval_type == "MedDistractQA-Nonliteral":
        cleaned_sentences_alpha = [s.strip('"') for s in CONFOUNDER_SENTENCES_ALPHA]
        main(args.engine, ["medqaNoOpAlphaGen"], [cleaned_sentences_alpha])

    elif args.eval_type == "MedDistractQA-Bystander":
        cleaned_sentences_beta = [s.strip('"') for s in CONFOUNDER_SENTENCES_BETA]
        main(args.engine, ["medqaNoOpBetaGen"], [cleaned_sentences_beta])
    
    else:
        print("No matched task.")
