#include "audio-projector.h"
#include "common-nexa.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vector>

struct ggml_tensor *audio_projector_inference(audio_projector &model, std::vector<float> &audio_feature_data)
{
    // Build the computation graph for inference
    struct ggml_cgraph *gf = model.build_graph();
    // Allocate the graph tensors
    ggml_gallocr_alloc_graph(model.compute_alloc, gf);

    // Set the input data
    struct ggml_tensor *input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, audio_feature_data.data(), 0, audio_feature_data.size() * sizeof(float));

    model.set_n_threads(0);

    // Execute the graph on the backend
    ggml_backend_graph_compute(model.backend, gf);

    // Return the output tensor (last node in the graph)
    return ggml_graph_get_tensor(gf, "output");
}

struct ggml_tensor *audio_projector_inference(audio_projector &model, struct ggml_tensor *audio_feature_tensor)
{
    // Set the input data
    std::vector<float> data(ggml_nelements(audio_feature_tensor));
    ggml_backend_tensor_get(audio_feature_tensor, data.data(), 0, ggml_nbytes(audio_feature_tensor));

    return audio_projector_inference(model, data);
}
