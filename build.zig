// Compatible with Zig Version 0.11.0
const std = @import("std");
const Compile = std.Build.Step.Compile;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.Mode;
const CrossTarget = std.zig.CrossTarget;

const Maker = struct {
    builder: *std.build.Builder,
    target: CrossTarget,
    optimize: Mode,
    config_header: *ConfigHeader,

    const cflags = .{"-std=c11"};
    const cxxflags = .{"-std=c++11"};

    fn init(builder: *std.build.Builder) Maker {
        const commit_hash = @embedFile(".git/refs/heads/master");
        const config_header = builder.addConfigHeader(
            .{ .style = .blank, .include_path = "build-info.h" },
            .{
                .BUILD_NUMBER = 0,
                .BUILD_COMMIT = commit_hash[0 .. commit_hash.len - 1], // omit newline
            },
        );
        return Maker{
            .builder = builder,
            .target = builder.standardTargetOptions(.{}),
            .optimize = builder.standardOptimizeOption(.{}),
            .config_header = config_header,
        };
    }

    fn obj(m: *const Maker, name: []const u8, src: []const u8) *Compile {
        const o = m.builder.addObject(.{ .name = name, .target = m.target, .optimize = m.optimize });
        if (std.mem.endsWith(u8, src, ".c")) {
            o.addCSourceFiles(&.{src}, &cflags);
            o.linkLibC();
        } else {
            o.addCSourceFiles(&.{src}, &cxxflags);
            o.linkLibCpp();
        }
        o.addIncludePath(.{ .path = "." });
        o.addIncludePath(.{ .path = "./examples" });
        return o;
    }

    fn exe(m: *const Maker, name: []const u8, src: []const u8, deps: []const *Compile) *Compile {
        const e = m.builder.addExecutable(.{ .name = name, .target = m.target, .optimize = m.optimize });
        e.addIncludePath(.{ .path = "." });
        e.addIncludePath(.{ .path = "./examples" });
        e.addCSourceFiles(&.{src}, &cxxflags);
        for (deps) |d| e.addObject(d);
        e.linkLibC();
        e.linkLibCpp();
        e.addConfigHeader(m.config_header);
        m.builder.installArtifact(e);

        // Currently a bug is preventing correct linking for optimized builds for Windows:
        // https://github.com/ziglang/zig/issues/15958
        if (e.target.isWindows()) {
            e.want_lto = false;
        }
        return e;
    }
};

pub fn build(b: *std.build.Builder) void {
    const make = Maker.init(b);

    const ggml = make.obj("ggml", "ggml.c");
    const ggml_alloc = make.obj("ggml-alloc", "ggml-alloc.c");
    const llama = make.obj("llama", "llama.cpp");
    const common = make.obj("common", "examples/common.cpp");
    const grammar_parser = make.obj("grammar-parser", "examples/grammar-parser.cpp");

    _ = make.exe("main", "examples/main/main.cpp", &.{ ggml, ggml_alloc, llama, common, grammar_parser });
    _ = make.exe("quantize", "examples/quantize/quantize.cpp", &.{ ggml, ggml_alloc, llama });
    _ = make.exe("perplexity", "examples/perplexity/perplexity.cpp", &.{ ggml, ggml_alloc, llama, common });
    _ = make.exe("embedding", "examples/embedding/embedding.cpp", &.{ ggml, ggml_alloc, llama, common });
    _ = make.exe("train-text-from-scratch", "examples/train-text-from-scratch/train-text-from-scratch.cpp", &.{ ggml, ggml_alloc, llama });

    const server = make.exe("server", "examples/server/server.cpp", &.{ ggml, ggml_alloc, llama, common, grammar_parser });
    if (server.target.isWindows()) {
        server.linkSystemLibrary("ws2_32");
    }
}
