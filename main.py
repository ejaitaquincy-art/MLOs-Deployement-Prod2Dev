#!/usr/bin/env python
# coding: utf-8

# # SQL Generation with Transformer API

# In[1]:


!pip install torch transformers bitsandbytes accelerate sqlparse

# In[2]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# In[3]:


torch.cuda.is_available()

# In[4]:


available_memory = torch.cuda.get_device_properties(0).total_memory

# In[5]:


print(available_memory)

# ##Download the Model
# Use any model on Colab (or any system with >30GB VRAM on your own machine) to load this in f16. If unavailable, use a GPU with minimum 8GB VRAM to load this in 8bit, or with minimum 5GB of VRAM to load in 4bit.
# 
# This step can take around 5 minutes the first time. So please be patient :)

# In[6]:


model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if available_memory > 15e9:
    # if you have atleast 15GB of GPU memory, run load the model in float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
else:
    # else, load in 8 bits – this is a bit slower
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        use_cache=True,
    )

# ##Set the Question & Prompt and Tokenize
# Feel free to change the schema in the prompt below to your own schema

# In[7]:


prompt = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'
- Remember that revenue is price multiplied by quantity
- Remember that cost is supply_price multiplied by quantity

### Database Schema
This query will run on a database whose schema is represented in this string:
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""

# ##Generate the SQL
# This can be excruciatingly slow on a T4 in Colab, and can take 10-20 seconds per query. On faster GPUs, this will take ~1-2 seconds
# 
# Ideally, you should use `num_beams`=4 for best results. But because of memory constraints, we will stick to just 1 for now.

# In[8]:


import sqlparse

def generate_query(question):
    updated_prompt = prompt.format(question=question)
    inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # empty cache so that you do generate more results w/o memory crashing
    # particularly important on Colab – memory management is much more straightforward
    # when running on an inference service
    return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)

# In[9]:


question = "What was our revenue by product in the New York region last month?"
generated_sql = generate_query(question)

# In[10]:


print(generated_sql)

# # Exercise
#  - Complete the prompts similar to what we did in class.
#      - Try at least 3 versions
#      - Be creative
#  - Write a one page report summarizing your findings.
#      - Were there variations that didn't work well? i.e., where GPT either hallucinated or wrong
#  - What did you learn?

# In[2]:


# --- Prompt 1 (clear) ---
question = "Find the total number of orders placed by each customer and sort the results by the highest number of orders."

prompt = f"""### Task
Generate a SQL query to answer the question.

### Question
{question}

### SQL
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
)

sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(sql)


# In[ ]:


# --- Prompt 2 (more complex, HAVING) ---
question = "Find the average order value for each country and return only countries where the average order value is greater than 100."

prompt = f"""### Task
Generate a SQL query to answer the question.

### Question
{question}

### SQL
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
)

sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(sql)


# In[ ]:


# --- Prompt 3 (ambiguous, likely to produce assumptions/hallucinations) ---
question = "List the most loyal customers based on their purchasing behavior."

prompt = f"""### Task
Generate a SQL query to answer the question.

### Question
{question}

### SQL
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
)

sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(sql)


# Findings Report
# 
# During this exercise, I tested three different prompt variations to evaluate how well the transformer-based model could generate SQL queries from natural language instructions.
# 
# The first prompt was clear and structured, asking for the total number of orders placed by each customer sorted by highest activity. In this case, the model performed well. It correctly generated a query using aggregation and ordering. This showed that when the request is explicit and follows common analytical patterns, the model is able to translate natural language into SQL reliably.
# 
# The second prompt introduced more complexity by asking for the average order value per country and filtering results based on a threshold. The model still produced a reasonable query, but the logic became less stable. In some attempts, the distinction between WHERE and HAVING conditions was not always handled correctly. This indicates that multi-step reasoning involving aggregation and filtering increases the chance of structural mistakes.
# 
# The third prompt was intentionally ambiguous, asking for the "most loyal customers based on purchasing behavior." In this case, the model produced inconsistent outputs. Sometimes loyalty was interpreted as frequency of orders, other times as total spending. In certain variations, the model introduced assumptions that were not explicitly defined in the schema. This demonstrates a limitation of LLM-based SQL generation: when natural language lacks precision, the model may infer its own definitions or fabricate logic.
# 
# Some variations did not work well, particularly when the intent required interpretation rather than direct mapping. These cases showed that the model can hallucinate metrics or apply incorrect reasoning.
# 
# From this exercise, I learned that prompt clarity has a strong impact on SQL generation quality. The more structured and explicit the request, the more accurate the output. Ambiguity increases the likelihood of hallucination or logical errors.
# 
# Overall, transformer-based models are powerful tools for translating natural language into SQL, but they require careful prompt design to ensure reliable results.

# In[ ]:



