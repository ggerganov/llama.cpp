const std = @import("std");

// Zig Version: 0.11.0-dev.3379+629f0d23b
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    lib.linkLibCpp();
    lib.addIncludePath(".");
    lib.addIncludePath("./examples");
    lib.addCSourceFiles(&.{
        "ggml.c",
    }, &.{"-std=c11"});
    lib.addCSourceFiles(&.{
        "llama.cpp",
    }, &.{"-std=c++11"});
    b.installArtifact(lib);

    const examples = .{
        "main",
        "baby-llama",
        "embedding",
        // "metal",
        "perplexity",
        "quantize",
        "quantize-stats",
        "save-load-state",
        // "server",
        "simple",
        "train-text-from-scratch",
    };

    inline for (examples) |example_name| {
        const exe = b.addExecutable(.{
            .name = example_name,
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath(".");
        exe.addIncludePath("./examples");
        exe.addCSourceFiles(&.{
            std.fmt.comptimePrint("examples/{s}/{s}.cpp", .{example_name, example_name}),
            "examples/common.cpp",
        }, &.{"-std=c++11"});
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_" ++ example_name, "Run the app");
        run_step.dependOn(&run_cmd.step);
    }
}
