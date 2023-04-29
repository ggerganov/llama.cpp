# llama.cpp/example/jeopardy

This is pretty much just a straight port of aigoopy/llm-jeopardy/ with an added graph viewer.

The jeopardy test can be used to compare the fact knowledge of different models and compare them to eachother. This is in contrast to some other tests, which test logical deduction, creativity, writing skills, etc.


Step 1: Open jeopardy.sh and modify the following:
```
MODEL=(path to your model)
MODEL_NAME=(name of your model)
prefix=(basically, if you use vicuna it's Human: , if you use something else it might be User: , etc)
opts=(add -instruct here if needed for your model, or anything else you want to test out)
```
Step 2: Run `jeopardy.sh` from the llama.cpp folder

Step 3: Repeat steps 1 and 2 until you have all the results you need.

Step 4: Run `graph.py`, and follow the instructions. At the end, it will generate your final graph.

Note: The Human bar is based off of the full, original 100 sample questions. If you modify the question count or questions, it will not be valid.
