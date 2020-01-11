#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# default_exp core


# # jetson-haiku
# 
# > API details.
# 

# In[ ]:


# -*- coding: utf-8 -*-

"""Top-level package for jetson-haiku, GPT2 haikubot for Jetson Nano"""

__author__ = """LemurTime"""
__email__ = "software@ananthropicprose.com"
__version__ = "0.0.1"

#Todo: Update with new GPT2 model
#Todo: fine-tine GPT2 output
#Todo: add logging
#Todo: add Twitter output?


# In[ ]:


#export

#define list of all haikus
global finished_haiku
global finished_haiku_list


# In[ ]:


#export

import syllapy
#import gpt2Pytorch as gp2py
#rather, let's just incorporate the gp2pytorch code, for now:


# In[ ]:


#export

#Initial arguments go here:

GPT2_seed_text="Cherry trees in the summer."
args_nsamples = 1
args_batch_size = -1
args_length = 1
args_unconditional = 0
args_temperature = 0.9
args_top_k = 40
args_quiet = 1
verse_input = ""


# In[ ]:


#export

#Rather than import GPT2, code here (need to update with new GPT2 model later) 
#Need to fix this, and import rather than keep GPT2 directory in folder

'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

def text_generator(state_dict):
   # parser = argparse.ArgumentParser()
  #  parser.add_argument("--text", type=str, required=True)
   # parser.add_argument("--quiet", type=bool, default=False)
   # parser.add_argument("--nsamples", type=int, default=1)
   # parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
   # parser.add_argument("--batch_size", type=int, default=-1)
   # parser.add_argument("--length", type=int, default=-1)
   # parser.add_argument("--temperature", type=float, default=0.7)
   # parser.add_argument("--top_k", type=int, default=40)
   # args = parser.parse_args()

    if args_quiet is False:
        print(args)

   # if args_batch_size == -1:
    args_batch_size = 1
    assert args_nsamples % args_batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    #if args_length == -1:
    args_length = config.n_ctx // 2
   # elif args_length > config.n_ctx:
    #    raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

   # print(args.text)
    context_tokens = enc.encode(GPT2_seed_text)

    generated = 0
    for _ in range(args_nsamples // args_batch_size):
        out = sample_sequence(
            model=model, length=args_length,
            context=context_tokens  if not  args_unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args_unconditional else None,
            batch_size=args_batch_size,
            temperature=args_temperature, top_k=args_top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args_batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args_quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            global GPT2_output
            GPT2_output = text
            print(text)

if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
     #   text_generator(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()


# In[ ]:


#export

##General verse_gen - input is input, number of syllables required

def verse_gen(verse_input, syllable_length):
    global verse_words
    global verse_string
    global verse_count
    global verse_syllable_count
    
    global verse_one_string

#Go to first whitespace, count syllables.  Continue until "syllable_length" syllables.  If over required amount syllables try with new input.
#initialize counter
    y=0
    x=1
    verse_syllable_count=0

#Split to remove whitespace
    verse_words=verse_input.split(' ')

    while verse_syllable_count < syllable_length:
        print("Adding next word to the string")

#Put the first word in a string
        verse_string=' '.join(verse_words[y:x])

#Count the syllables
        verse_syllable_count = syllapy.count(verse_string)
    
#increment x
        x=x+1

#Get new input if the words don't make 5 syllables
#        if verse_syllable_count > syllable_length:
 #           print("Need new input")
  #          text_generator(state_dict)
   #         verse_input = GPT2_output
    #        verse_gen(verse_input, syllable_length)
        
#If the words make 5 syllables, check for period or comma at the end of it.  Use if so, get new input if not       
     #   if verse_syllable_count == syllable_length:
          #  if verse_string[-1] == "." or verse_string[-1] == ",":
           #     print(verse_string)
        #    else:
         #       print("Need input ending with punctuation")
      #         verse_gen(verse_input, syllable_length)
            
        
        
## New way:  go down the input to look for haiku-able phrases.  If not, get new input

        if verse_syllable_count == syllable_length:
            print(verse_string)
            return verse_string
    
        if verse_syllable_count > syllable_length:
        #reinitialize the string and keep going
            print("Moving up in string")
            print(verse_string)
            
            #reinitialize verse_string
            verse_string=""
            verse_syllable_count=0
            y=x-1
            
            #verse_gen(verse_input, syllable_length)

       
      

#END OF VERSE ONE GEN    



##Now we will take verse_one_string, and add it to our haiku




        


# In[ ]:


#export

## Code to run the module

def haiku_gen():
    text_generator(state_dict)

#Code to generate verse 1:
    verse_string = ""
    verse_input = GPT2_output
    syllable_length = 5
    verse_one_string=verse_gen(verse_input, syllable_length)
    
#Code to generate verse 2:
    verse_string = ""
    GPT2_seed_text = verse_one_string
    text_generator(state_dict)
    verse_input = GPT2_output
    syllable_length = 7
    verse_two_string=verse_gen(verse_input, syllable_length)

#Code to generate verse 3:
    verse_string = ""
    GPT2_seed_text = verse_one_string
    text_generator(state_dict)
    verse_input = GPT2_output
    syllable_length=5
    verse_three_string=verse_gen(verse_input, syllable_length)

#Print finished haiku
    print("Here is the haiku:")

#Print finished haiku
 
    finished_haiku=''
    finished_haiku='\n'.join([verse_one_string,verse_two_string,verse_three_string])
    print(finished_haiku)
    
    #Add finished haiku to a list
    f = open("haikulist.txt", "a")
    f.write("\n\n"+finished_haiku)
    f.close()
    
    #Place finished haiku in an input for GUI (clear it out first)
    f = open("latesthaiku.txt", "w")
    f.write(finished_haiku)
    f.close()


    #Put verse2 in as GPT seedtext seed
    f = open("haikuseed.txt", "w")
    f.write(verse_two_string)
    f.close()
    
    from IPython.display import Audio

    wave = np.sin(2*np.pi*400*np.arange(10000*2)/10000)
    Audio(wave, rate=30000, autoplay=True)
    
 


# In[ ]:


#export

##Run the module:

#Initial arguments go here:
from IPython.display import Audio
GPT2_seed_text="Gorillas in the mist."
args_nsamples = 1
args_batch_size = -1
args_length = 1
args_unconditional = 0
args_temperature = 0.9
args_top_k = 40
args_quiet = 1
verse_input = ""
z = 0

while z < 100:
    
    haiku_gen()
    f = open(r"haikuseed.txt")
    GPT2_seed_text=f.readline()
    f.close()
    
    
    #Beep after each iteration
    wave = np.sin(2*np.pi*400*np.arange(10000*2)/10000)
    Audio(wave, rate=30000, autoplay=True)
    z+1
    
#Beep when all done

wave = np.sin(2*np.pi*400*np.arange(10000*2)/10000)
Audio(wave, rate=40000, autoplay=True)


# In[ ]:


#hide
#from nbdev.showdoc import *


# In[ ]:


#from nbdev.export import *
#notebook2script()


# In[ ]:


#Todo:
#Training feature - what is a good/bad haiku


# In[ ]:



##Todo: make ananthropic
#Remove: wordlist indicating persons or personification (I, his, hers, mine, ours, who)


