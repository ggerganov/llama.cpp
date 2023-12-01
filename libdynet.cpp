//#include <format>
#define FMT_HEADER_ONLY 1
#include <fmt/core.h>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <iostream>

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


size_t BATCH_SIZE=1000;
static vector<vector<float>> batch(BATCH_SIZE);
static int next_id = 0;
static int batch_id = 0;
void ggml_tensor_add(const char * name,const struct ggml_tensor * tensor);

#include <eigen3/Eigen/Core>
//Eigen/Core
using namespace Eigen;

void ggml_tensor_add(const char * name,const struct ggml_tensor * tensor){
  //model.add(tensor)
  //runtime2::debug(std::cout,e);
  //runtime2::debug(std::cout,tensor);

  const size_t num_elements = ggml_nelements(tensor);
  if (tensor->type != GGML_TYPE_F32) {
      return;
    }
  float* buffer = ggml_get_data_f32(tensor);
  //Expression x = input(cg,  buffer);
  //  runtime2::debug(std::cout,x);
  std::vector<float> values;

  // copy the elements  in
  std::copy(buffer, &buffer[num_elements], back_inserter(values));

  batch[(next_id++) % BATCH_SIZE] = values;

  if ((next_id) % BATCH_SIZE == 0)
  {
    batch_id ++;
    ofstream data_file;      // pay attention here! ofstream

    data_file.open(fmt::format("batch{}.bin", batch_id), ios::out | ios::binary);
    
    for (auto &row: batch) {

      auto bsize = row.size();
      data_file.write(reinterpret_cast<char*>(&bsize),4);
      
      for (auto &cell: row) {
	data_file.write(reinterpret_cast<char*>(&cell), 4);
      }
    }
    data_file.close();
  }

  //trainmain();
  //runtime2::debug(std::cout,batch);
}

void init_dynet() {

}

class DynetWrapper{

  void init(){

  }
};
 

//ComputationGraph cg2;


const unsigned HIDDEN_SIZE = 8;

int ITERATIONS = 5;

void trainmain() {

//   char**  argv = 0;
//   //= {""};
//   int argc = 0;
//   dynet::initialize(argc,argv);
//     static SimpleSGDTrainer trainer(model);
// Parameter p_W = model.add_parameters({HIDDEN_SIZE, 2});
// Parameter p_b = model.add_parameters({HIDDEN_SIZE});
// Parameter p_V = model.add_parameters({1, HIDDEN_SIZE});
// Parameter p_a = model.add_parameters({1});

// Expression W = parameter(cg, p_W);
// Expression b = parameter(cg, p_b);
// Expression V = parameter(cg, p_V);
// Expression a = parameter(cg, p_a);

//   // Train the parameters.
//   for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
//     double loss = 0;
//     for (unsigned mi = 0; mi < BATCH_SIZE; ++mi) {
      
//       auto x_values = batch[mi];
//       //auto y_value = x_values.batch_ptr(0);

//       Expression y = input(cg, y_value);

//       Expression x = input(cg, x_values.batch_ptr(0));
//       Expression h = tanh(W*x + b);
//       Expression y_pred = V*h + a;
//       Expression loss_expr = squared_distance(y_pred, y);

//       loss += as_scalar(cg.forward(loss_expr));
//       cg.backward(loss_expr);
//       trainer.update();
//     }
//     loss /= 4;
//     cerr << "E = " << loss << endl;
//   }

}





#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#ifdef BOOST_REGEX
  #include <boost/regex.hpp>
  using namespace boost;
#else
  #include <regex>
#endif

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <dynet/lstm.h>

using namespace std;
using namespace std::chrono;
using namespace dynet;

// Read a file where each line is of the form "word1|tag1 word2|tag2 ..."
// Yields pairs of lists of the form < [word1, word2, ...], [tag1, tag2, ...] >
vector<pair<vector<string>, vector<string> > > read(const string & fname) {
  ifstream fh(fname);
  if(!fh) throw std::runtime_error("Could not open file");
  string str;
  regex re("[ |]");
  vector<pair<vector<string>, vector<string> > > sents;
  while(getline(fh, str)) {
    pair<vector<string>,vector<string> > word_tags;
    sregex_token_iterator first{str.begin(), str.end(), re, -1}, last;
    while(first != last) {
      word_tags.first.push_back(*first++);
      assert(first != last);
      word_tags.second.push_back(*first++);
    }
    sents.push_back(word_tags);
  }
  return sents;
}

class BiLSTMTagger {
public:

  BiLSTMTagger(unsigned layers, unsigned wembed_dim, unsigned hidden_dim, unsigned mlp_dim, ParameterCollection & model, Dict & wv, Dict & tv, unordered_map<string,int> & wc)
                        : wv(wv), tv(tv), wc(wc) {
    unsigned nwords = wv.size();
    unsigned ntags  = tv.size();
    word_lookup = model.add_lookup_parameters(nwords, {wembed_dim});

    // MLP on top of biLSTM outputs 100 -> 32 -> ntags
    pH = model.add_parameters({mlp_dim, hidden_dim*2});
    pO = model.add_parameters({ntags, mlp_dim});

    // word-level LSTMs
    fwdRNN = VanillaLSTMBuilder(layers, wembed_dim, hidden_dim, model); // layers, in-dim, out-dim, model
    bwdRNN = VanillaLSTMBuilder(layers, wembed_dim, hidden_dim, model);
  }

  Dict &wv, &tv;
  unordered_map<string,int> & wc;
  LookupParameter word_lookup;
  Parameter pH, pO;
  VanillaLSTMBuilder fwdRNN, bwdRNN;

  // Do word representation
  Expression word_rep(ComputationGraph & cg, const string & w) {
    return lookup(cg, word_lookup, wv.convert(wc[w] > 5 ? w : "<unk>"));
  }

  vector<Expression> build_tagging_graph(ComputationGraph & cg, const vector<string> & words) {
    // parameters -> expressions
    Expression H = parameter(cg, pH);
    Expression O = parameter(cg, pO);

    // initialize the RNNs
    fwdRNN.new_graph(cg);
    bwdRNN.new_graph(cg);

    // get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    vector<Expression> wembs(words.size()), fwds(words.size()), bwds(words.size()), fbwds(words.size());
    for(size_t i = 0; i < words.size(); ++i)
      wembs[i] = word_rep(cg, words[i]);

    // feed word vectors into biLSTM
    fwdRNN.start_new_sequence();
    for(size_t i = 0; i < wembs.size(); ++i)
      fwds[i] = fwdRNN.add_input(wembs[i]);
    bwdRNN.start_new_sequence();
    for(size_t i = wembs.size(); i > 0; --i)
      bwds[i-1] = bwdRNN.add_input(wembs[i-1]);

    // Concatenate and MLP
    for(size_t i = 0; i < wembs.size(); ++i)
      fbwds[i] = O * tanh( H * concatenate({fwds[i], bwds[i]}) );

    return fbwds;
  }

  Expression sent_loss(ComputationGraph & cg, vector<string> & words, vector<string> & tags) {
    vector<Expression> exprs = build_tagging_graph(cg, words), errs(words.size());
    for(size_t i = 0; i < tags.size(); ++i)
      errs[i] = pickneglogsoftmax(exprs[i], tv.convert(tags[i]));
    return sum(errs);
  }

  vector<string> tag_sent(vector<string> & words) {
    ComputationGraph cg;
    vector<Expression> exprs = build_tagging_graph(cg, words), errs(words.size());
    vector<string> tags(words.size());
    for(size_t i = 0; i < words.size(); ++i) {
      vector<float> scores = as_vector(exprs[i].value());
      size_t max_id = distance(scores.begin(), max_element(scores.begin(), scores.end()));
      tags[i] = tv.convert(max_id);
    }
    return tags;
  }

};

int othermain() {
  int argc=0;
  char**argv=0;
  time_point<system_clock> start = system_clock::now();

  vector<pair<vector<string>, vector<string> > > train = read("data/tags/train.txt");
  vector<pair<vector<string>, vector<string> > > dev = read("data/tags/dev.txt");
  Dict word_voc, tag_voc;
  unordered_map<string, int> word_cnt;
  for(auto & sent : train) {
    for(auto & w : sent.first) {
      word_voc.convert(w);
      word_cnt[w]++;
    }
    for(auto & t : sent.second)
      tag_voc.convert(t);
  }
  tag_voc.freeze();
  word_voc.convert("<unk>"); word_voc.freeze(); word_voc.set_unk("<unk>");

  // DyNet Starts
  dynet::initialize(argc, argv);
  ParameterCollection model;
  AdamTrainer trainer(model);
  trainer.clipping_enabled = false;

  if(argc != 6) {
    cerr << "Usage: " << argv[0] << " WEMBED_SIZE HIDDEN_SIZE MLP_SIZE SPARSE TIMEOUT" << endl;
    return 1;
  }
  int WEMBED_SIZE = atoi(argv[1]);
  int HIDDEN_SIZE = atoi(argv[2]);
  int MLP_SIZE = atoi(argv[3]);
  trainer.sparse_updates_enabled = atoi(argv[4]);
  int TIMEOUT = atoi(argv[5]);

  // Initilaize the tagger
  BiLSTMTagger tagger(1, WEMBED_SIZE, HIDDEN_SIZE, MLP_SIZE, model, word_voc, tag_voc, word_cnt);

  {
    duration<float> fs = (system_clock::now() - start);
    float startup_time = duration_cast<milliseconds>(fs).count() / float(1000);
    cout << "startup time: " << startup_time << endl;
  }

  // Do training
  start = system_clock::now();
  int i = 0, all_tagged = 0, this_words = 0;
  float this_loss = 0.f, all_time = 0.f;
  for(int iter = 0; iter < 100; iter++) {
    shuffle(train.begin(), train.end(), *dynet::rndeng);
    for(auto & s : train) {
      i++;
      if(i % 500 == 0) {
        trainer.status();
        cout << this_loss/this_words << endl;
        all_tagged += this_words;
        this_loss = 0.f;
        this_words = 0;
      }
      if(i % 10000 == 0) {
        duration<float> fs = (system_clock::now() - start);
        all_time += duration_cast<milliseconds>(fs).count() / float(1000);
        int dev_words = 0, dev_good = 0;
        float dev_loss = 0;
        for(auto & sent : dev) {
          vector<string> tags = tagger.tag_sent(sent.first);
          for(size_t j = 0; j < tags.size(); ++j)
            if(tags[j] == sent.second[j])
              dev_good++;
          dev_words += sent.second.size();
        }
        cout << "acc=" << dev_good/float(dev_words) << ", time=" << all_time << ", word_per_sec=" << all_tagged/all_time << endl;
        if(all_time > TIMEOUT)
          exit(0);
        start = system_clock::now();
      }

      ComputationGraph cg;
      Expression loss_exp = tagger.sent_loss(cg, s.first, s.second);
      float my_loss = as_scalar(cg.forward(loss_exp));
      this_loss += my_loss;
      this_words += s.first.size();
      cg.backward(loss_exp);
      trainer.update();
    }
  }
  return 0;
}
