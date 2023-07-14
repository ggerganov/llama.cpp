const std = @import("std");
const commit_hash = @embedFile(".git/refs/heads/master");

// Zig Version: 0.11.0-dev.3986+e05c242cd
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const config_header = b.addConfigHeader(
        .{ .style = .blank, .include_path = "build-info.h" },
        .{
            .BUILD_NUMBER = 0,
            .BUILD_COMMIT = commit_hash[0 .. commit_hash.len - 1], // omit newline
        },
    );

    const lib = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    lib.linkLibCpp();
    lib.addIncludePath(".");
    lib.addIncludePath("./examples");
    lib.addConfigHeader(config_header);
    lib.addCSourceFiles(&.{"ggml.c"}, &.{"-std=c11"});
    lib.addCSourceFiles(&.{"llama.cpp"}, &.{"-std=c++11"});
    b.installArtifact(lib);

    const examples = .{
        "main",
        "baby-llama",
        "embedding",
        "metal",
        "perplexity",
        "quantize",
        "quantize-stats",
        "save-load-state",
        "server",
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
        exe.addConfigHeader(config_header);
        exe.addCSourceFiles(&.{
            std.fmt.comptimePrint("examples/{s}/{s}.cpp", .{ example_name, example_name }),
            "examples/common.cpp",
        }, &.{"-std=c++11"});
        exe.linkLibrary(lib);
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);

        const run_step = b.step("run-" ++ example_name, "Run the app");
        run_step.dependOn(&run_cmd.step);
    }
}
