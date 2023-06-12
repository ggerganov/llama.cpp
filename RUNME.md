# Running the Actual model

This is the best input that we were able to get the best output of NER from the vicuna model.

Start with this:

- Act as a food entity recognizer. Try to recognize the food entities in the following sentence the user will give. Do not re-write the sentence again. Write your answer as: [recognized food entities] => ['item1', 'item2', 'item3']. Do not explain your answer.

- Can you recognize the food-related entities in the following sentence: “I want to eat a burger with fries and a coke from Ibn-ElSham”?

- Can you recognize the food-related entities in the following sentence: “I would like to order chicken ranch pizza from papa johns without olives”?





## Command

To run the model to just get the output out of one single prompt

```bash
make -j && ./main -m /home/g1-s23/dev/Models/vicuna-ggml-vic13b-q4_0.bin -p "Hello! Can you tell me what is the capital of Egypt?" -n 128
```

To run the model in instruct mode and interactive

```bash
make -j && ./main -m /home/g1-s23/dev/Models/vicuna-ggml-vic13b-q4_0.bin --instruct -n 128 -i
```
