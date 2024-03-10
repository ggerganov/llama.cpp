// Compatible with Zig Version 0.12.0-dev.xxxx
const std = @import("std");
const ArrayList = std.ArrayList;
const Compile = std.Build.Step.Compile;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.OptimizeMode;
const Target = std.Build.ResolvedTarget;

const Maker = struct {
    builder: *std.Build,
    target: Target,
    optimize: Mode,
    enable_lto: bool,

    include_dirs: ArrayList([]const u8),
    cflags: ArrayList([]const u8),
    cxxflags: ArrayList([]const u8),

    fn addInclude(m: *Maker, dir: []const u8) !void {
        try m.include_dirs.append(dir);
    }
    fn addProjectInclude(m: *Maker, path: []const []const u8) !void {
        try m.addInclude(try m.builder.build_root.join(m.builder.allocator, path));
    }
    fn addCFlag(m: *Maker, flag: []const u8) !void {
        try m.cflags.append(flag);
    }
    fn addCxxFlag(m: *Maker, flag: []const u8) !void {
        try m.cxxflags.append(flag);
    }
    fn addFlag(m: *Maker, flag: []const u8) !void {
        try m.addCFlag(flag);
        try m.addCxxFlag(flag);
    }

    fn init(builder: *std.Build) !Maker {
        const target = builder.standardTargetOptions(.{});
        const zig_version = @import("builtin").zig_version_string;
        const commit_hash = try std.ChildProcess.run(
            .{ .allocator = builder.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        );
        try std.fs.cwd().writeFile("common/build-info.cpp", builder.fmt(
            \\int LLAMA_BUILD_NUMBER = {};
            \\char const *LLAMA_COMMIT = "{s}";
            \\char const *LLAMA_COMPILER = "Zig {s}";
            \\char const *LLAMA_BUILD_TARGET = "{s}";
            \\
        , .{ 0, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, try target.query.zigTriple(builder.allocator) }));

        var m = Maker{
            .builder = builder,
            .target = target,
            .optimize = builder.standardOptimizeOption(.{}),
            .enable_lto = false,
            .include_dirs = ArrayList([]const u8).init(builder.allocator),
            .cflags = ArrayList([]const u8).init(builder.allocator),
            .cxxflags = ArrayList([]const u8).init(builder.allocator),
        };

        try m.addCFlag("-std=c11");
        try m.addCxxFlag("-std=c++11");

        if (m.target.result.abi == .gnu) {
            try m.addFlag("-D_GNU_SOURCE");
        }
        if (m.target.result.os.tag == .macos) {
            try m.addFlag("-D_DARWIN_C_SOURCE");
        }
        try m.addFlag("-D_XOPEN_SOURCE=600");

        if (m.target.result.abi == .gnu) {
            try m.addFlag("-D_GNU_SOURCE");
        }
        if (m.target.result.os.tag == .macos) {
            try m.addFlag("-D_DARWIN_C_SOURCE");
        }
        try m.addFlag("-D_XOPEN_SOURCE=600");

        try m.addProjectInclude(&.{});
        try m.addProjectInclude(&.{"common"});
        return m;
    }

    fn obj(m: *const Maker, name: []const u8, src: []const u8) *Compile {
        const o = m.builder.addObject(.{ .name = name, .target = m.target, .optimize = m.optimize });

        if (std.mem.endsWith(u8, src, ".c") or std.mem.endsWith(u8, src, ".m")) {
            o.addCSourceFiles(.{ .files = &.{src}, .flags = m.cflags.items });
            o.linkLibC();
        } else {
            o.addCSourceFiles(.{ .files = &.{src}, .flags = m.cxxflags.items });
            if (m.target.result.abi == .msvc) {
                o.linkLibC(); // need winsdk + crt
            } else {
                // linkLibCpp already add (libc++ + libunwind + libc)
                o.linkLibCpp();
            }
        }
        for (m.include_dirs.items) |i| o.addIncludePath(.{ .path = i });
        o.want_lto = m.enable_lto;
        return o;
    }

    fn exe(m: *const Maker, name: []const u8, src: []const u8, deps: []const *Compile) *Compile {
        const e = m.builder.addExecutable(.{ .name = name, .target = m.target, .optimize = m.optimize });
        e.addCSourceFiles(.{ .files = &.{src}, .flags = m.cxxflags.items });
        for (deps) |d| e.addObject(d);
        for (m.include_dirs.items) |i| e.addIncludePath(.{ .path = i });

        // https://github.com/ziglang/zig/issues/15448
        if (m.target.result.abi == .msvc) {
            e.linkLibC(); // need winsdk + crt
        } else {
            // linkLibCpp already add (libc++ + libunwind + libc)
            e.linkLibCpp();
        }
        m.builder.installArtifact(e);
        e.want_lto = m.enable_lto;

        const run = m.builder.addRunArtifact(e);
        run.step.dependOn(m.builder.getInstallStep());
        if (m.builder.args) |args| {
            run.addArgs(args);
        }
        const step = m.builder.step(name, m.builder.fmt("Run the {s} example", .{name}));
        step.dependOn(&run.step);

        return e;
    }
};

pub fn build(b: *std.Build) !void {
    var make = try Maker.init(b);
    make.enable_lto = b.option(bool, "lto", "Enable LTO optimization, (default: false)") orelse false;

    // Options
    const llama_vulkan = b.option(bool, "llama-vulkan", "Enable Vulkan backend for Llama, (default: false)") orelse false;
    const llama_metal = b.option(bool, "llama-metal", "Enable Metal backend for Llama, (default: false, true for macos)") orelse (make.target.result.os.tag == .macos);
    const llama_no_accelerate = b.option(bool, "llama-no-accelerate", "Disable Accelerate framework for Llama, (default: false)") orelse false;
    const llama_accelerate = !llama_no_accelerate and make.target.result.os.tag == .macos;

    // Flags
    if (llama_accelerate) {
        try make.addFlag("-DGGML_USE_ACCELERATE");
        try make.addFlag("-DACCELERATE_USE_LAPACK");
        try make.addFlag("-DACCELERATE_LAPACK_ILP64");
    }

    // Objects
    var extras = ArrayList(*Compile).init(b.allocator);

    if (llama_vulkan) {
        try make.addFlag("-DGGML_USE_VULKAN");
        const ggml_vulkan = make.obj("ggml-vulkan", "ggml-vulkan.cpp");
        try extras.append(ggml_vulkan);
    }

    if (llama_metal) {
        try make.addFlag("-DGGML_USE_METAL");
        const ggml_metal = make.obj("ggml-metal", "ggml-metal.m");
        try extras.append(ggml_metal);
    }

    const ggml = make.obj("ggml", "ggml.c");
    const ggml_alloc = make.obj("ggml-alloc", "ggml-alloc.c");
    const ggml_backend = make.obj("ggml-backend", "ggml-backend.c");
    const ggml_quants = make.obj("ggml-quants", "ggml-quants.c");
    const llama = make.obj("llama", "llama.cpp");
    const buildinfo = make.obj("common", "common/build-info.cpp");
    const common = make.obj("common", "common/common.cpp");
    const console = make.obj("console", "common/console.cpp");
    const sampling = make.obj("sampling", "common/sampling.cpp");
    const grammar_parser = make.obj("grammar-parser", "common/grammar-parser.cpp");
    const clip = make.obj("clip", "examples/llava/clip.cpp");
    const train = make.obj("train", "common/train.cpp");
    const llava = make.obj("llava", "examples/llava/llava.cpp");

    // Executables
    const server = make.exe("server", "examples/server/server.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo, sampling, console, grammar_parser, clip, llava });
    if (make.target.result.os.tag == .windows) {
        server.linkSystemLibrary("ws2_32");
    }

    const exes = [_]*Compile{
        make.exe("main", "examples/main/main.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo, sampling, console, grammar_parser, clip }),
        make.exe("simple", "examples/simple/simple.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo, sampling, console, grammar_parser, clip }),
        make.exe("quantize", "examples/quantize/quantize.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo }),
        make.exe("perplexity", "examples/perplexity/perplexity.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo }),
        make.exe("embedding", "examples/embedding/embedding.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo }),
        make.exe("finetune", "examples/finetune/finetune.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo, train }),
        make.exe("train-text-from-scratch", "examples/train-text-from-scratch/train-text-from-scratch.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo, train }),
        make.exe("parallel", "examples/parallel/parallel.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, common, buildinfo, sampling, console, grammar_parser, clip }),
        server,
    };

    for (exes) |e| {
        for (extras.items) |o| e.addObject(o);

        if (llama_vulkan) {
            e.linkSystemLibrary("vulkan");
        }

        if (llama_metal) {
            e.linkFramework("Foundation");
            e.linkFramework("Metal");
            e.linkFramework("MetalKit");
        }

        if (llama_accelerate) {
            e.linkFramework("Accelerate");
        }
    }
}
