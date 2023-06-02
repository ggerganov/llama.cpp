# Token generation performance tips

## Verifying that the model is running on the GPU
Make sure you compiled llama with the correct env variables according to [this guide](../README.md#cublas)

When running `llama.cpp`, outputs some helpful diagnostic information to stderr.
To verify that the workload is
