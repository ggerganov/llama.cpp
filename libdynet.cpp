#include <iostream>
#include <type_traits>

using namespace std;
#include <refl-cpp/refl.hpp>

// now for some old school c++ hack
#define private public

#include "llama.h"
#include "llama-internal.hpp"
#include "print.hpp"

#include <dynet/tensor.h>

#include "dynet/training.h"
#include "dynet/expr.h"
//#include "dynet/io.h"
#include "dynet/model.h"
using namespace dynet;

// FILE /usr/local/include/dynet/except.h
REFL_TYPE(dynet::out_of_memory)
REFL_END
// FILE /usr/local/include/dynet/except.h
REFL_TYPE(dynet::cuda_not_implemented)
REFL_END
// FILE /usr/local/include/dynet/except.h
REFL_TYPE(dynet::cuda_exception)
REFL_END
// FILE /usr/local/include/dynet/dim.h
REFL_TYPE(dynet::Dim)
  REFL_FIELD(d)
  REFL_FIELD(nd)
  REFL_FIELD(bd)
REFL_END
// FILE /usr/local/include/dynet/mem.h
REFL_TYPE(dynet::MemAllocator)
  REFL_FIELD(align)
REFL_END
// FILE /usr/local/include/dynet/mem.h
REFL_TYPE(dynet::CPUAllocator)
REFL_END
// FILE /usr/local/include/dynet/mem.h
REFL_TYPE(dynet::SharedAllocator)
REFL_END
// FILE /usr/local/include/dynet/globals.h
REFL_TYPE(dynet::Device)
REFL_END
// FILE /usr/local/include/dynet/globals.h
REFL_TYPE(dynet::NamedTimer)
REFL_END
// FILE /usr/local/include/dynet/aligned-mem-pool.h
REFL_TYPE(dynet::InternalMemoryPool)
  REFL_FIELD(used)
  REFL_FIELD(name)
  REFL_FIELD(capacity)
  REFL_FIELD(a)
  REFL_FIELD(mem)
REFL_END
// FILE /usr/local/include/dynet/aligned-mem-pool.h
REFL_TYPE(dynet::AlignedMemoryPool)
  REFL_FIELD(name)
  REFL_FIELD(pools)
  REFL_FIELD(cap)
  REFL_FIELD(current)
  REFL_FIELD(a)
  REFL_FIELD(expanding_unit)
REFL_END

REFL_TYPE(dynet::DeviceMempoolSizes)
  REFL_FIELD(used)
REFL_END
// FILE /usr/local/include/dynet/tensor.h
REFL_TYPE(dynet::IndexTensor)
REFL_END
// FILE /usr/local/include/dynet/tensor.h
REFL_TYPE(dynet::Tensor)
  REFL_FIELD(d)
  REFL_FIELD(v)
  REFL_FIELD(device)
  REFL_FIELD(mem_pool)
REFL_END
// FILE /usr/local/include/dynet/tensor.h
REFL_TYPE(dynet::TensorTools)
REFL_END
// FILE /usr/local/include/dynet/weight-decay.h
REFL_TYPE(dynet::L2WeightDecay)
  REFL_FIELD(weight_decay)
  REFL_FIELD(lambda)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::DeviceManager)
REFL_END

// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::ParameterInit)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::ParameterStorageBase)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::ParameterStorage)
  REFL_FIELD(name)
  REFL_FIELD(dim)
  REFL_FIELD(values)
  REFL_FIELD(g)
  REFL_FIELD(updated)
  REFL_FIELD(nonzero_grad)
  REFL_FIELD(owner)
  REFL_FIELD(device)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::ParameterStorageCreator)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::LookupParameterStorage)
  REFL_FIELD(name)
  REFL_FIELD(all_dim)
  REFL_FIELD(all_values)
  REFL_FIELD(all_grads)
  REFL_FIELD(dim)
  REFL_FIELD(values)
  REFL_FIELD(grads)
  REFL_FIELD(non_zero_grads)
  REFL_FIELD(updated)
  REFL_FIELD(all_updated)
  REFL_FIELD(nonzero_grad)
  REFL_FIELD(owner)
  REFL_FIELD(device)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::LookupParameterStorageCreator)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::Parameter)
  REFL_FIELD(p)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::LookupParameter)
  REFL_FIELD(p)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::ParameterCollectionStorage)
  REFL_FIELD(all_params)
  REFL_FIELD(params)
  REFL_FIELD(lookup_params)
  REFL_FIELD(gradient_norm_scratch)
  REFL_FIELD(weight_decay)
  REFL_FIELD(device_manager)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::ParameterCollection)
REFL_FIELD(name) //get_fullname
  REFL_FIELD(name_cntr)
  REFL_FIELD(collec_name_cntr)
  REFL_FIELD(storage)
  REFL_FIELD(parent)
REFL_END
// FILE /usr/local/include/dynet/model.h
REFL_TYPE(dynet::Model)
REFL_END



// FILE /usr/local/include/dynet/shadow-params.h
REFL_TYPE(dynet::ShadowParameters)
  REFL_FIELD(h)
REFL_END
// FILE /usr/local/include/dynet/shadow-params.h
REFL_TYPE(dynet::ShadowLookupParameters)
  REFL_FIELD(all_h)
  REFL_FIELD(h)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::Trainer)
  REFL_FIELD(learning_rate)
  REFL_FIELD(clipping_enabled)
  REFL_FIELD(clip_threshold)
  REFL_FIELD(clips)
  REFL_FIELD(updates)
  REFL_FIELD(clips_since_status)
  REFL_FIELD(updates_since_status)
  REFL_FIELD(sparse_updates_enabled)
  REFL_FIELD(aux_allocated)
  REFL_FIELD(aux_allocated_lookup)
  REFL_FIELD(ema_beta)
  REFL_FIELD(ma_mode)
  REFL_FIELD(ma_params_swapped)
  REFL_FIELD(ma_params_saved)
  REFL_FIELD(ma_update_freq)
  REFL_FIELD(ma_updates)
  REFL_FIELD(ma_aux_allocated)
  REFL_FIELD(ma_aux_allocated_lookup)
  REFL_FIELD(ma_p)
  REFL_FIELD(ma_lp)
  REFL_FIELD(ma_saved_p)
  REFL_FIELD(ma_saved_lp)
  REFL_FIELD(model)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::SimpleSGDTrainer)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::CyclicalSGDTrainer)
  REFL_FIELD(e_min)
  REFL_FIELD(e_max)
  REFL_FIELD(step_size)
  REFL_FIELD(gamma)
  REFL_FIELD(it)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::MomentumSGDTrainer)
  REFL_FIELD(vp)
  REFL_FIELD(vlp)
  REFL_FIELD(momentum)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::AdagradTrainer)
  REFL_FIELD(epsilon)
  REFL_FIELD(vp)
  REFL_FIELD(vlp)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::AdadeltaTrainer)
  REFL_FIELD(epsilon)
  REFL_FIELD(rho)
  REFL_FIELD(hg)
  REFL_FIELD(hlg)
  REFL_FIELD(hd)
  REFL_FIELD(hld)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::RMSPropTrainer)
  REFL_FIELD(epsilon)
  REFL_FIELD(rho)
  REFL_FIELD(hmsg)
  REFL_FIELD(hlmsg)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::AdamTrainer)
  REFL_FIELD(beta_1)
  REFL_FIELD(beta_2)
  REFL_FIELD(epsilon)
  REFL_FIELD(m)
  REFL_FIELD(lm)
  REFL_FIELD(v)
  REFL_FIELD(lv)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::AmsgradTrainer)
  REFL_FIELD(beta_1)
  REFL_FIELD(beta_2)
  REFL_FIELD(epsilon)
  REFL_FIELD(m)
  REFL_FIELD(lm)
  REFL_FIELD(v)
  REFL_FIELD(lv)
  REFL_FIELD(vhat)
  REFL_FIELD(lvhat)
REFL_END
// FILE /usr/local/include/dynet/training.h
REFL_TYPE(dynet::EGTrainer)
  REFL_FIELD(momentum)
  REFL_FIELD(hp)
  REFL_FIELD(hlp)
  REFL_FIELD(e_min)
  REFL_FIELD(e_max)
  REFL_FIELD(step_size)
  REFL_FIELD(gamma)
  REFL_FIELD(it)
  REFL_FIELD(isCyclical)
  REFL_FIELD(zeg)
  REFL_FIELD(meg)
REFL_END
// FILE /usr/local/include/dynet/init.h
REFL_TYPE(dynet::DynetParams)
  REFL_FIELD(random_seed)
  REFL_FIELD(mem_descriptor)
  REFL_FIELD(weight_decay)
  REFL_FIELD(autobatch)
  REFL_FIELD(profiling)
  REFL_FIELD(shared_parameters)
  REFL_FIELD(ngpus_requested)
  REFL_FIELD(ids_requested)
  REFL_FIELD(cpu_requested)
  REFL_FIELD(requested_gpus)
  REFL_FIELD(gpu_mask)
REFL_END
// FILE /usr/local/include/dynet/sig.h
REFL_TYPE(dynet::SigYoav)
  REFL_FIELD(which)
  REFL_FIELD(nn)
  REFL_FIELD(nd)
  REFL_FIELD(dims)
  REFL_FIELD(node_ids)
REFL_END
// FILE /usr/local/include/dynet/sig.h
REFL_TYPE(dynet::SigString)
  REFL_FIELD(which)
  REFL_FIELD(data)
  REFL_FIELD(tail)
REFL_END
// FILE /usr/local/include/dynet/sig.h
REFL_TYPE(dynet::SigHash)
  REFL_FIELD(hash)
  REFL_FIELD(which)
REFL_END
// FILE /usr/local/include/dynet/sig.h
REFL_TYPE(dynet::SigHasher)
REFL_END
// FILE /usr/local/include/dynet/dynet.h
REFL_TYPE(dynet::ExecutionEngine)
REFL_END
// FILE /usr/local/include/dynet/dynet.h
REFL_TYPE(dynet::ParameterNodeBase)
REFL_END



// FILE /usr/local/include/dynet/dynet.h
REFL_TYPE(dynet::CGCheckpoint)
  REFL_FIELD(node_idx)
  REFL_FIELD(par_node_idx)
  REFL_FIELD(device_mem_checkpoint)
REFL_END
// FILE /usr/local/include/dynet/dynet.h
REFL_TYPE(dynet::ComputationGraph)
  REFL_FIELD(nodes)
  REFL_FIELD(parameter_nodes)
  REFL_FIELD(ee)
  REFL_FIELD(graph_id)
  REFL_FIELD(immediate_compute)
  REFL_FIELD(check_validity)
  REFL_FIELD(checkpoints)
REFL_END
// FILE /usr/local/include/dynet/dynet.h
REFL_TYPE(dynet::Node)
  REFL_FIELD(args)
  REFL_FIELD(dim)
  REFL_FIELD(device)
  REFL_FIELD(forward_inplace_state)
  REFL_FIELD(backward_inplace_state)
  REFL_FIELD(cg_)
  REFL_FIELD(aux_mem)
  REFL_FIELD(has_cuda_implemented)
REFL_END
// FILE /usr/local/include/dynet/expr.h
REFL_TYPE(dynet::Expression)
//REFL_FIELD(pg) computation graph
  REFL_FIELD(i)
  REFL_FIELD(graph_id)
REFL_END

// FILE /usr/local/include/dynet/io.h
//REFL_TYPE(dynet::Saver) 
//REFL_END
// FILE /usr/local/include/dynet/io.h
//REFL_TYPE(dynet::Loader)
//REFL_END
// FILE /usr/local/include/dynet/io.h
//REFL_TYPE(dynet::TextFileSaver)
//  REFL_FIELD(p_datastream)
//  REFL_FIELD(datastream)
//REFL_END
// FILE /usr/local/include/dynet/io.h
//REFL_TYPE(dynet::TextFileLoader)
//  REFL_FIELD(dataname)
//REFL_END

void trainmain();

static ParameterCollection  model;
size_t BATCH_SIZE=500;
static  ComputationGraph cg;
static vector<Expression> batch(BATCH_SIZE);
static int next_id = 0;
void ggml_tensor_add(const char * name,const struct ggml_tensor * tensor);
void ggml_tensor_add(const char * name,const struct ggml_tensor * tensor){
  //model.add(tensor)
  //runtime2::debug(std::cout,e);
  //runtime2::debug(std::cout,tensor);

  const size_t num_elements = ggml_nelements(tensor);
  if (tensor->type != GGML_TYPE_F32) {
      return;
    }
  float* buffer = ggml_get_data_f32(tensor);
  Expression x = input(cg,  buffer);

  runtime2::debug(std::cout,x);
  batch[(next_id++) % BATCH_SIZE] = x;

  //runtime2::debug(std::cout,batch);
}

void trainmain() {

  int argc=0;
  char** argv;
  dynet::initialize(argc, argv);
  const unsigned ITERATIONS = 30;
  //ParameterCollection m;
  SimpleSGDTrainer trainer(model);

  const unsigned HIDDEN_SIZE = 8;
  Parameter p_W = model.add_parameters({HIDDEN_SIZE, 2});
  Parameter p_b = model.add_parameters({HIDDEN_SIZE});
  Parameter p_V = model.add_parameters({1, HIDDEN_SIZE});
  Parameter p_a = model.add_parameters({1});
  //if (argc == 2) {
    // Load the model and parameters from file if given.
    //TextFileLoader loader(argv[1]);
  //loader.populate(m);
    //}

  // Static declaration of the computation graph.
  ComputationGraph cg;
  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  // Set x_values to change the inputs to the network.
  vector<dynet::real> x_values(2);
  Expression x = input(cg, {2}, &x_values);
  dynet::real y_value;  // Set y_value to change the target output.
  Expression y = input(cg, &y_value);

  Expression h = tanh(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss_expr = squared_distance(y_pred, y);

  // Show the computation graph, just for fun.
  cg.print_graphviz();

  // Train the parameters.
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      y_value = (x1 != x2) ? 1 : -1;
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer.update();
    }
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  // Check whether our ComputationGraph learns correctly or not.
  x_values[0] = -1;	// Set input value
  x_values[1] = -1; // Set input value
  cg.forward(y_pred); // Calculate until y_pred node
  std::cout << "[-1,-1] -1 : " << as_scalar(y_pred.value()) << std::endl;
  x_values[0] = -1;
  x_values[1] =  1;
  cg.forward(y_pred);
  std::cout << "[-1, 1]  1 : " << as_scalar(y_pred.value()) << std::endl;
  x_values[0] =  1;
  x_values[1] = -1;
  cg.forward(y_pred);
  std::cout << "[ 1,-1]  1 : " << as_scalar(y_pred.value()) << std::endl;
  x_values[0] =  1;
  x_values[1] =  1;
  cg.forward(y_pred);
  std::cout << "[ 1, 1] -1 : " << as_scalar(y_pred.value()) << std::endl;

  // Output the model and parameter objects to a file.
  // TextFileSaver saver("test.model");
  //  saver.save(model);
  runtime2::debug(std::cout,model);
}

