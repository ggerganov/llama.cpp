// Compatible with Zig Version 0.11.0
const std = @import("std");
const ArrayList = std.ArrayList;
const Compile = std.Build.Step.Compile;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.Mode;
const CrossTarget = std.zig.CrossTarget;

const Maker = struct {
    builder: *std.build.Builder,
    target: CrossTarget,
    optimize: Mode,
    enable_lto: bool,

    include_dirs: ArrayList([]const u8),
    cflags: ArrayList([]const u8),
    cxxflags: ArrayList([]const u8),
    objs: ArrayList(*Compile),

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

    fn init(builder: *std.build.Builder) !Maker {
        const target = builder.standardTargetOptions(.{});
        const zig_version = @import("builtin").zig_version_string;
        const commit_hash = try std.ChildProcess.exec(
            .{ .allocator = builder.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        );
        try std.fs.cwd().writeFile("common/build-info.cpp", builder.fmt(
            \\int LLAMA_BUILD_NUMBER = {};
            \\char const *LLAMA_COMMIT = "{s}";
            \\char const *LLAMA_COMPILER = "Zig {s}";
            \\char const *LLAMA_BUILD_TARGET = "{s}";
            \\
        , .{ 0, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, try target.allocDescription(builder.allocator) }));
        var m = Maker{
            .builder = builder,
            .target = target,
            .optimize = builder.standardOptimizeOption(.{}),
            .enable_lto = false,
            .include_dirs = ArrayList([]const u8).init(builder.allocator),
            .cflags = ArrayList([]const u8).init(builder.allocator),
            .cxxflags = ArrayList([]const u8).init(builder.allocator),
            .objs = ArrayList(*Compile).init(builder.allocator),
        };

        try m.addCFlag("-std=c11");
        try m.addCxxFlag("-std=c++11");
        try m.addProjectInclude(&.{});
        try m.addProjectInclude(&.{"common"});
        return m;
    }

    fn obj(m: *const Maker, name: []const u8, src: []const u8) *Compile {
        const o = m.builder.addObject(.{ .name = name, .target = m.target, .optimize = m.optimize });
        if (o.target.getAbi() != .msvc)
            o.defineCMacro("_GNU_SOURCE", null);

        if (std.mem.endsWith(u8, src, ".c")) {
            o.addCSourceFiles(&.{src}, m.cflags.items);
            o.linkLibC();
        } else {
            o.addCSourceFiles(&.{src}, m.cxxflags.items);
            if (o.target.getAbi() == .msvc) {
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
        e.addCSourceFiles(&.{src}, m.cxxflags.items);
        for (deps) |d| e.addObject(d);
        for (m.objs.items) |o| e.addObject(o);
        for (m.include_dirs.items) |i| e.addIncludePath(.{ .path = i });

        // https://github.com/ziglang/zig/issues/15448
        if (e.target.getAbi() == .msvc) {
            e.linkLibC(); // need winsdk + crt
        } else {
            // linkLibCpp already add (libc++ + libunwind + libc)
            e.linkLibCpp();
        }
        m.builder.installArtifact(e);
        e.want_lto = m.enable_lto;
        return e;
    }
};

pub fn build(b: *std.build.Builder) !void {
    var make = try Maker.init(b);
    make.enable_lto = b.option(bool, "lto", "Enable LTO optimization, (default: false)") orelse false;

    const ggml = make.obj("ggml", "ggml.c");
    const ggml_alloc = make.obj("ggml-alloc", "ggml-alloc.c");
    const ggml_backend = make.obj("ggml-backend", "ggml-backend.c");
    const ggml_quants = make.obj("ggml-quants", "ggml-quants.c");
    const unicode = make.obj("unicode", "unicode.cpp");
    const llama = make.obj("llama", "llama.cpp");
    const buildinfo = make.obj("common", "common/build-info.cpp");
    const common = make.obj("common", "common/common.cpp");
    const console = make.obj("console", "common/console.cpp");
    const sampling = make.obj("sampling", "common/sampling.cpp");
    const grammar_parser = make.obj("grammar-parser", "common/grammar-parser.cpp");
    const train = make.obj("train", "common/train.cpp");
    const clip = make.obj("clip", "examples/llava/clip.cpp");
    const llava = make.obj("llava", "examples/llava/llava.cpp");

    _ = make.exe("main", "examples/main/main.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo, sampling, console, grammar_parser });
    _ = make.exe("quantize", "examples/quantize/quantize.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo });
    _ = make.exe("perplexity", "examples/perplexity/perplexity.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo });
    _ = make.exe("embedding", "examples/embedding/embedding.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo });
    _ = make.exe("finetune", "examples/finetune/finetune.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo, train });
    _ = make.exe("train-text-from-scratch", "examples/train-text-from-scratch/train-text-from-scratch.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo, train });

    const server = make.exe("server", "examples/server/server.cpp", &.{ ggml, ggml_alloc, ggml_backend, ggml_quants, llama, unicode, common, buildinfo, sampling, grammar_parser, clip, llava });
    if (server.target.isWindows()) {
        server.linkSystemLibrary("ws2_32");
    }
}
