load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


#new_local_repository(
#   name = "cuda",
#    path = "/usr/local/cuda/",
#    build_file = "cuda.BUILD",
#)


http_archive(
    name = "catch2",
    urls = ["https://github.com/catchorg/Catch2/archive/v3.4.0.tar.gz"],
    strip_prefix = "Catch2-3.4.0",
    sha256 = "122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb"
)

# bazel query "@vcpkg-build//include:*"
# bazel query 'labels(hdrs, ...)'
#local_repository(
#    name = "vcpkg-build",
#    path = "./build/vcpkg_installed/x64-linux",
#)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
# RUN (remember to build first):
# bazel build ... --config linux
# bazel run @hedron_compile_commands//:refresh_all -- --config linux
# OR...
# bazel run :refresh_compile_commands

# ADD SKYLIB: requirement for catch2
http_archive(
    name = "bazel_skylib",
    sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "aspect_bazel_lib",
    sha256 = "09b51a9957adc56c905a2c980d6eb06f04beb1d85c665b467f659871403cf423",
    strip_prefix = "bazel-lib-1.34.5",
    url = "https://github.com/aspect-build/bazel-lib/releases/download/v1.34.5/bazel-lib-v1.34.5.tar.gz",
)

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies")

aspect_bazel_lib_dependencies()


http_archive(
    name = "rules_cuda",
    #sha256 = "{sha256_to_replace}",
    strip_prefix = "rules_cuda-1a2ec3d1ffacf3c462b69c2bbac91111d1752d21",
    urls = ["https://github.com/bazel-contrib/rules_cuda/archive/1a2ec3d1ffacf3c462b69c2bbac91111d1752d21.tar.gz"],
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()