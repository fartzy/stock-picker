"""
Custom Bazel macro for running pytest tests.

This macro wraps py_test to provide pytest integration with:
- Automatic pytest_shim.py injection (eliminates run_tests(__file__) boilerplate)
- Timezone configuration for test determinism
- Default test size of "small" for faster scheduling
"""

load("@rules_python//python:defs.bzl", "py_test")

def py_pytest(name, srcs, deps = [], args = [], **kwargs):
    """
    Run Python tests using pytest.

    Args:
        name: Test target name
        srcs: List of test source files (must be explicit, no glob)
        deps: List of dependencies
        args: Additional arguments to pass to pytest
        **kwargs: Additional py_test arguments (env, tags, timeout, size, etc.)
    """
    env = kwargs.get("env", {})
    if "TZ" not in env:
        env["TZ"] = "UTC"
    kwargs["env"] = env

    if "size" not in kwargs:
        kwargs["size"] = "small"

    shim_label = Label("//bazel/rules/pytest:pytest_shim.py")

    final_deps = deps + ["@pypi//pytest"]

    py_test(
        name = name,
        srcs = [shim_label] + srcs,
        main = shim_label,
        args = args,
        deps = final_deps,
        **kwargs
    )
