# Ernie's documentation

```sh
pip install -r requirements.txt
make build    # the site into ../site/
make serve    # or preview it at http://127.0.0.1:8000
make clean
```

Build Ernie with Python into `build/` first: the Python API reference reads the module's stubs there. The C++ API reference runs Doxygen on the public headers (mkdoxy), so `doxygen` must be installed.
