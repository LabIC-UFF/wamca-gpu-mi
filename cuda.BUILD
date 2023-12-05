load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

# DOCUMENTATION ON SELECT:
# https://bazel.build/reference/be/functions?hl=pt-br#select

# MPIR instructions
# https://stackoverflow.com/questions/47359417/how-to-compile-gmp-for-windows-using-visual-studio

# thirdparty CUDA library
cc_library(
    name = "cuda",
    hdrs = glob(["include/**/*"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)
