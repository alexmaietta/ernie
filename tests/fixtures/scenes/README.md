Scene files for `json_test.cpp` and `schema_check.py`, written by hand:

- `rover.json`: a small scene touching each family of schemas (transforms in every form, a mesh, curves, an instancer, lights, a camera with an API schema, a material binding, a collection, render settings and custom attributes on a grouping). It must load, validate and pass the generated JSON Schema with no problems.
- `rover.saved.json`: what `SaveScene` writes for `rover.json`, compared byte for byte (golden).
- `invalid/problems.json`: one prim per kind of mistake `ernie validate` reports; `invalid/problems.expected` is its report, one problem per line (golden).

Golden files are rewritten by running the scene tests with `ERNIE_UPDATE_GOLDEN=1`; review the diff before committing it.
