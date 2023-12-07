#ifdef NDEBUG
#undef NDEBUG
#endif
#include <node/node.h>
#include "uv.h"
#include <assert.h>

#include <algorithm>

// Note: This file is being referred to from doc/api/embedding.md, and excerpts
// from it are included in the documentation. Try to keep these in sync.
// Snapshot support is not part of the embedder API docs yet due to its
// experimental nature, although it is of course documented in node.h.

using node::CommonEnvironmentSetup;
using node::Environment;
using node::MultiIsolatePlatform;
using v8::Context;
using v8::HandleScope;
using v8::Isolate;
using v8::Locker;
using v8::MaybeLocal;
using v8::V8;
using v8::Value;

static int RunNodeInstance(MultiIsolatePlatform* platform,
                           const std::vector<std::string>& args,
                           const std::vector<std::string>& exec_args);

std::string process_output_plugin(const std::string start,
				  const std::string state,
				  const std::string input) {
  
  //int main(int argc, char** argv) {
  //argv = uv_setup_args(argc, argv);
  std::vector<std::string> args;
  std::unique_ptr<node::InitializationResult> result =
      node::InitializeOncePerProcess(
          args,
          {node::ProcessInitializationFlags::kNoInitializeV8,
           node::ProcessInitializationFlags::kNoInitializeNodeV8Platform});

  for (const std::string& error : result->errors())
    fprintf(stderr, "%s: %s\n", args[0].c_str(), error.c_str());
  if (result->early_return() != 0) {
    return "ERROR";
  }

  std::unique_ptr<MultiIsolatePlatform> platform =
      MultiIsolatePlatform::Create(4);
  V8::InitializePlatform(platform.get());
  V8::Initialize();

  int ret =
      RunNodeInstance(platform.get(), result->args(), result->exec_args());

  V8::Dispose();
  V8::DisposePlatform();

  node::TearDownOncePerProcess();
  return "TODO";
}

int RunNodeInstance(MultiIsolatePlatform* platform,
                    const std::vector<std::string>& args,
                    const std::vector<std::string>& exec_args) {
  int exit_code = 0;

  // Format of the arguments of this binary:
  // Building snapshot:
  // embedtest js_code_to_eval arg1 arg2...
  //           --embedder-snapshot-blob blob-path
  //           --embedder-snapshot-create
  //           [--embedder-snapshot-as-file]
  // Running snapshot:
  // embedtest --embedder-snapshot-blob blob-path
  //           [--embedder-snapshot-as-file]
  //           arg1 arg2...
  // No snapshot:
  // embedtest arg1 arg2...
  node::EmbedderSnapshotData::Pointer snapshot;

  std::string binary_path = args[0];
  std::vector<std::string> filtered_args;
  bool is_building_snapshot = false;
  bool snapshot_as_file = false;
  std::string snapshot_blob_path;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "--embedder-snapshot-create") {
      is_building_snapshot = true;
    } else if (arg == "--embedder-snapshot-as-file") {
      snapshot_as_file = true;
    } else if (arg == "--embedder-snapshot-blob") {
      assert(i + 1 < args.size());
      snapshot_blob_path = args[i + 1];
      i++;
    } else {
      filtered_args.push_back(arg);
    }
  }

  if (!snapshot_blob_path.empty() && !is_building_snapshot) {
    FILE* fp = fopen(snapshot_blob_path.c_str(), "r");
    assert(fp != nullptr);
    if (snapshot_as_file) {
      snapshot = node::EmbedderSnapshotData::FromFile(fp);
    } else {
      uv_fs_t req = uv_fs_t();
      int statret =
          uv_fs_stat(nullptr, &req, snapshot_blob_path.c_str(), nullptr);
      assert(statret == 0);
      size_t filesize = req.statbuf.st_size;
      uv_fs_req_cleanup(&req);

      std::vector<char> vec(filesize);
      size_t read = fread(vec.data(), filesize, 1, fp);
      assert(read == 1);
      snapshot = node::EmbedderSnapshotData::FromBlob(vec);
    }
    assert(snapshot);
    int ret = fclose(fp);
    assert(ret == 0);
  }

  if (is_building_snapshot) {
    // It contains at least the binary path, the code to snapshot,
    // and --embedder-snapshot-create (which is filtered, so at least
    // 2 arguments should remain after filtering).
    assert(filtered_args.size() >= 2);
    // Insert an anonymous filename as process.argv[1].
    filtered_args.insert(filtered_args.begin() + 1,
                         node::GetAnonymousMainPath());
  }

  std::vector<std::string> errors;
  std::unique_ptr<CommonEnvironmentSetup> setup =
      snapshot
          ? CommonEnvironmentSetup::CreateFromSnapshot(
                platform, &errors, snapshot.get(), filtered_args, exec_args)
      : is_building_snapshot ? CommonEnvironmentSetup::CreateForSnapshotting(
                                   platform, &errors, filtered_args, exec_args)
                             : CommonEnvironmentSetup::Create(
                                   platform, &errors, filtered_args, exec_args);
  if (!setup) {
    for (const std::string& err : errors)
      fprintf(stderr, "%s: %s\n", binary_path.c_str(), err.c_str());
    return 1;
  }

  Isolate* isolate = setup->isolate();
  Environment* env = setup->env();

  {
    Locker locker(isolate);
    Isolate::Scope isolate_scope(isolate);
    HandleScope handle_scope(isolate);
    Context::Scope context_scope(setup->context());

    MaybeLocal<Value> loadenv_ret;
    if (snapshot) {  // Deserializing snapshot
      loadenv_ret = node::LoadEnvironment(env, node::StartExecutionCallback{});
    } else if (is_building_snapshot) {
      // Environment created for snapshotting must set process.argv[1] to
      // the name of the main script, which was inserted above.
      loadenv_ret = node::LoadEnvironment(
          env,
          "const assert = require('assert');"
          "assert(require('v8').startupSnapshot.isBuildingSnapshot());"
          "globalThis.embedVars = { nön_ascıı: '🏳️‍🌈' };"
          "globalThis.require = require;"
          "require('vm').runInThisContext(process.argv[2]);");
    } else {
      loadenv_ret = node::LoadEnvironment(
          env,
          "const publicRequire = require('module').createRequire(process.cwd() "
          "+ '/');"
          "globalThis.require = publicRequire;"
          "globalThis.embedVars = { nön_ascıı: '🏳️‍🌈' };"
          "require('vm').runInThisContext(process.argv[1]);");
    }

    if (loadenv_ret.IsEmpty())  // There has been a JS exception.
      return 1;

    exit_code = node::SpinEventLoop(env).FromMaybe(1);
  }

  if (!snapshot_blob_path.empty() && is_building_snapshot) {
    snapshot = setup->CreateSnapshot();
    assert(snapshot);

    FILE* fp = fopen(snapshot_blob_path.c_str(), "w");
    assert(fp != nullptr);
    if (snapshot_as_file) {
      snapshot->ToFile(fp);
    } else {
      const std::vector<char> vec = snapshot->ToBlob();
      size_t written = fwrite(vec.data(), vec.size(), 1, fp);
      assert(written == 1);
    }
    int ret = fclose(fp);
    assert(ret == 0);
  }

  node::Stop(env);

  return exit_code;
}
