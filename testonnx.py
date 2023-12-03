
# Certainly! To use ONNX Runtime to load and run a model in Python, you'll need to follow these steps:

# 1. **Install ONNX Runtime:**
#    You can install ONNX Runtime using pip:

#    ```bash
#    pip install onnxruntime
#    ```

# 2. **Load the ONNX Model:**
#    Load your ONNX model using the `onnxruntime.InferenceSession` class:

#    ```python
#    import onnxruntime

#    # Replace 'your_model.onnx' with the path to your ONNX model

# 3. **Prepare Input Data:**
#    Prepare input data as a dictionary where keys are the input names specified in your ONNX model and values are the corresponding input data:

#    ```python
#    import numpy as np

#    # Replace 'input_name' with the actual input name from your model
#    input_name = session.get_inputs()[0].name
#    input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)

#    input_dict = {input_name: input_data}
#    ```

# 4. **Run Inference:**
#    Run the inference using the `run` method of the `InferenceSession`:

#    ```python
#    output = session.run(None, input_dict)
#    ```

#    Replace `None` with the names of the output nodes in your model if you want to retrieve specific outputs.

# Here's a condensed version of the above steps:

# ```python
import onnxruntime
import numpy as np

model_path = 'build/model.onnx'
session = onnxruntime.InferenceSession(model_path)

#{'name': 'embedding_input', 'type': {'tensorType': {'elemType': 1, 'shape': {'dim': [{'dimParam': 'unk__261'}, {'dimParam': 'unk__262'}]}}}}
input_name = "embedding_input"
#3session.get_inputs()[0].name

input_data = np.zeros((4096,2),dtype=np.float32)
for x in (range(2)):
    output = session.run(None, {input_name: input_data})
    #print(input_data.shape,output.shape)
    print("in",input_data.size)
    print("out",len(output[0]))
    print("in","".join([ str(j) for j in input_data[0:16]]))
    print("out","".join([ str(j) for j in output[0][0:16]]))

    for i,x in enumerate(output[0]):
        input_data[i] = x
        #input_data[4096 + i] = x


# Remember to replace `'your_model.onnx'` with the actual path to your ONNX model, and adjust input data accordingly based on your model's input requirements.
