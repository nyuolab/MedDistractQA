prompt_eval_mod = f'''
## Question
{{question}}

## Choices
{{choices}}

## Context
{{context}}

## Instructions
You are provided a medical multiple choice Question and potentially some medical Context. Solve the question in a step-by-step fashion and provide a single letter as the final answer in the end, formatted as "Therefore, the final answer is [X]."
'''

## Prompts for generating confounding sentences:

prompt_gen_alpha_confounder = f"""
Example sentences (for other questions):

1. "The patient's zodiac sign is Cancer."
2. "The patient said their friend's business is hemorrhaging money after a bad investment."
3. "The patient mentioned that their new car has a transmission fluid leak."
4. "The patient described their favorite art exhibit as having vascular pattern in the sculptures."
5. "The patient's professor delivered a lecture about the heart of ancient civilizations."

## Medical Question: {{question}}

## Clinical topic: {{clinical_topic}}

Generate one sentence using a medical term related to the clinical topic, where clinical terminology is used as nonmedical meaning. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.

Only respond with the confounding sentence.
"""
    
prompt_gen_beta_confounder = f"""
Example sentences:

1. "The patient's sister said her friend has high blood pressure and has been having seizures."
2. "The patient's dog has diabetes and cannot see properly."
3. "The patient's neighbor mentioned that their cat has been sneezing and refusing to eat."
4. "The patient's cousin said her coworker's father has arthritis and struggles to walk long distances."
5. "The patient's brother mentioned that his friend's goldfish has been swimming erratically and losing color."
    

## Medical Question: {{question}}

## Clinical topic: {{clinical_topic}}

Generate one statement, where another person/animal is referenced with relation to a medical term as seen in the example sentences. The statement should also be related to the clinical topic. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.

Only respond with the confounding sentence.
"""

prompt_select_system_topic = f"""

## Question
{{question}}

## Choices
{{choices}}

You are a medical expert. Please select the topic of the question from the following list of topics:

1. Human Development
2. Immune System
3. Blood & Lymphoreticular System
4. Behavioral Health
5. Nervous System & Special Senses	
6. Musculoskeletal System/Skin & Subcutaneous Tissue	
7. Cardiovascular System	
8. Respiratory System	
9. Gastrointestinal System	
10. Renal & Urinary System & Reproductive Systems	
11. Pregnancy, Childbirth & the Puerperium	
12. Endocrine System	
13. Multisystem Processes & Disorders	
14. Biostatistics & Epidemiology/Population Health/Interpretation of Medical Literature	
15. Social Sciences: Communication and Interpersonal Skills	
16. Social Sciences: Legal/Ethical Issues & Professionalism/Systems-based Practice & Patient Safety

ONLY RESPOND WITH THE TOPIC NUMBER (1-15). Do not include any other text in your response. 
"""


prompt_select_competency_topic = f"""

## Question
{{question}}

## Choices
{{choices}}

1. Medical Knowledge/Scientific Concepts (Applying foundational science concepts)
2. Patient Care: Diagnosis  
3. Patient Care: Management
4. Communication
5. Professionalism, Including Legal and Ethical Issues
6. Systems-based Practice, Including Patient Safety
7. Practice-based Learning

Only respond with the topic number (1-7) that BEST fits with the medical question. Do not include any other text in your response. 

"""
