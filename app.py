import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from llama_ccp import Llama
# import os
# import multiprocessing
import time

### Silence errors
##os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"
##os.environ["LLAMA_SET_ROWS"] = "1"
##
##NUM_THREADS = multiprocessing.cpu_count()  # usually 12–16 for Intel 125H
##
### === Load DeepSeek Chat GGUF model ===
##llm = Llama(
##    model_path=r"C:\Users\Sted\Documents\Ateneo\Y5S0\CSCI 217\20250711\deepseek-llm-7b-chat.Q4_K_M.gguf",
##    n_ctx=4096,
##    n_threads=NUM_THREADS,
##    use_mlock=True,
##    n_batch=64,
##    verbose=False
##)

def count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8")))

def build_prompt(history):
    prompt = "<|system|>\nYou are a helpful assistant.\n"
    for user_msg, assistant_msg in history:
        prompt += f"<|user|>\n{user_msg.strip()}\n<|assistant|>\n{assistant_msg.strip()}\n"
    return prompt

available_tokens = 2000

defaults = {
    'conversations' : [],
    'active' : 1,
    'input_tokens' : [0],
    'output_tokens' : [0],
    'total_tokens' : [0],
    'input_overflow' : 0,
    'output_quality' : 2,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

##col1, col2 = st.columns([0.8, 0.2])
##with col1:
##    st.write('Chats')
##with col2:
##    st.write('Tokens Used')

for i in range(len(st.session_state.conversations)):
##    col1, col2 = st.columns([0.8, 0.2])
##    with col1:
##        with st.chat_message('user'):
##            st.write(st.session_state.conversations[i][0])
##    with col2:
##        st.write(st.session_state.input_tokens[i])
##    col1, col2 = st.columns([0.8, 0.2])
##    with col1:
##        with st.chat_message('ai'):
##            st.write(st.session_state.conversations[i][1])
##    with col2:
##        st.write(st.session_state.output_tokens[i])
    with st.chat_message('user'):
        st.write(st.session_state.conversations[i][0])
    with st.chat_message('ai'):
        st.write(st.session_state.conversations[i][1])

if st.session_state.active == 1:
    
    user_input = st.chat_input('What\'s on your mind?')
    if user_input is not None:
        if user_input.lower() in ['quit', 'stop', 'exit']:
            st.session_state.active = 0
            st.write('Thank you for chatting')
            st.write(f'In this chat, a total of {st.session_state.total_tokens[-1]} have been used')
            st.write(f'You have {available_tokens-st.session_state.total_tokens[-1]} remaining')
            
    if user_input and st.session_state.active == 1:
        st.session_state.conversations.append((user_input, ''))
        # prompt = build_prompt(conversation[:-1]) + f"<|user|>\n{user_input}\n<|assistant|>\n"
        prompt = user_input # temporary replacement
        # input_tokens = count_tokens(prompt)
        input_tokens = random.randint(64, 128) # temporary replacement
        
        if input_tokens > available_tokens - st.session_state.total_tokens[-1]:
            st.session_state.input_overflow = 1
            st.session_state.conversations = st.session_state.conversations[:-1] # remove
            st.write(f'''Your input '{user_input}' corresponds to too many tokens.\n
Available:   {available_tokens}\n
Total: {st.session_state.total_tokens[-1]}\n
Input: {input_tokens}''')

        if st.session_state.input_overflow == 0:
            st.session_state.input_tokens.append(input_tokens)
            st.session_state.total_tokens.append(st.session_state.total_tokens[-1] + input_tokens)
            start = time.time()

##            col1, col2 = st.columns([0.8, 0.2])
##            with col1:
##                with st.chat_message('user'):
##                    st.write(prompt)
##            with col2:
##                st.write(input_tokens)

            with st.chat_message('user'):
                st.write(prompt)

            min_tokens = 32

            if available_tokens - st.session_state.total_tokens[-1] >= 256:
                st.session_state.output_quality = 2 # high
                # output = llm(prompt, max_tokens=256), stop=["<|user|>", "</s>"])
                # response = output["choices"][0]["text"].strip()
                response = 'Hello 2' # temporary replacement
                st.session_state.conversations[-1] = (user_input, response)
                # output_tokens = count_tokens(response)
                output_tokens = random.randint(32, 256) # temporary replacement
            elif available_tokens - st.session_state.total_tokens[-1] >= 32:
                st.session_state.output_quality = 1 # medium
                # output = llm(prompt, max_tokens=available_tokens - st.session_state.total_tokens[-1]), stop=["<|user|>", "</s>"])
                # response = output["choices"][0]["text"].strip()
                response = 'Hello 1'
                st.write('[Note: Output may be incomplete due to low token count.]') # temporary replacement
                st.session_state.conversations[-1] = (user_input, response)
                # output_tokens = count_tokens(response)
                output_tokens = random.randint(32, available_tokens - st.session_state.total_tokens[-1]) # temporary replacement
            else:
                st.session_state.output_quality = 0 # cannot output anything
                # response = ''
                response = 'Hello 0'
                st.write('[Note: Not enough tokens to write an output]') # temporary replacement
                st.session_state.conversations[-1] = (user_input, '')
                # output_tokens = count_tokens(response)
                output_tokens = 0 # temporary replacement
            
            st.session_state.output_tokens.append(output_tokens)
            elapsed = time.time() - start

##            col1, col2 = st.columns([0.8, 0.2])
##            with col1:
##                with st.chat_message('ai'):
##                    st.write(response)
##            with col2:
##                st.write(output_tokens)

            with st.chat_message('ai'):
                st.write(response)
            st.session_state.total_tokens[-1] += output_tokens
            st.write(f'A total of {st.session_state.total_tokens[-1]} have been used')

data = [
    st.session_state.input_tokens[1:],
    st.session_state.output_tokens[1:],
    st.session_state.total_tokens[1:]
]
df = pd.DataFrame(data, ['Input', 'Output', 'Total']).T
# st.write(df)
            
with st.sidebar:
    if st.session_state.active == 1:
        st.write('Chat is active')
    else:
        st.write('Chat is inactive')
    if st.session_state.input_overflow == 1:
        st.write('Input: Too many tokens')
    else:
        st.write('Input: No issues')
    if st.session_state.output_quality == 2:
        st.write('Output: High quality')
    elif st.session_state.output_quality == 1:
        st.write('Output: Medium quality')
    else:
        st.write('Output: Lack of tokens')
    
    st.write(f'Available tokens: {available_tokens}')
    st.write(f'Input tokens: {st.session_state.input_tokens[-1]}')
    st.write(f'Output tokens: {st.session_state.output_tokens[-1]}')
    percentage = f'{st.session_state.total_tokens[-1] / available_tokens:.2%}' 
    st.write(f'Total tokens: {st.session_state.total_tokens[-1]} ({percentage} used)')
    
    fig, ax = plt.subplots()
    ax.plot(np.array(df.index) + 1, df['Total'])
    ax.set_xlabel('Conversation Number')
    ax.set_ylabel('Tokens Used')
    ax.set_title('Tokens Used Over Time')
    ax.set_ylim(bottom = 0, top = available_tokens * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(y = available_tokens, color = 'red', linestyle = '--')
    ax.grid(True)
    if len(st.session_state.output_tokens) >= 3:
        st.pyplot(fig)
