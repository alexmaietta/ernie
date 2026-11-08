# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ernie._ernie import Prim, Scene, pipeline_node


def _port(node: Prim, side: str) -> str:
    info = pipeline_node(node["id"])
    if info is None:
        raise ValueError(f"{node.path} runs {node['id']}, which isn't a sensor pipeline node")
    ports = [name for name, (_, optional) in info[side].items() if not optional]
    if len(ports) != 1:
        raise ValueError(f"{node.path} has {side} {sorted(info[side])}, so name the port to connect")
    return ports[0]


def _consumers(pipeline: Prim, node: Prim, output: str) -> Iterator[tuple[Prim, str]]:
    for prim in [pipeline, *pipeline.children]:
        for port, source, connected in prim.connections:
            if source == node.path and connected == output:
                yield prim, port


def define_pipeline(scene: Scene, path: str) -> Prim:
    "Defines an empty SensorPipeline at path."
    return scene.define(path, "SensorPipeline")


def add_node(
    pipeline: Prim,
    name: str,
    node_id: str,
    input: Prim | None = None,
    inputs: Mapping[str, Prim | tuple[Prim, str]] | None = None,
    **values: Any,
) -> Prim:
    "Adds a node named name running node_id, fed by input's only output or by inputs port by port."
    node = pipeline.scene.define(f"{pipeline.path}/{name}", "Shader", {"id": node_id, **values})
    if input is not None:
        node.connect(_port(node, "inputs"), input, _port(input, "outputs"))
    for port, source in (inputs or {}).items():
        prim, output = source if isinstance(source, tuple) else (source, _port(source, "outputs"))
        node.connect(port, prim, output)
    return node


def connect_output(pipeline: Prim, name: str, node: Prim, output: str | None = None) -> None:
    "Makes node's output, its only one by default, the pipeline output that render vars read as name."
    pipeline.connect(name, node, output or _port(node, "outputs"))


def insert_after(pipeline: Prim, node: Prim, name: str, node_id: str, **values: Any) -> Prim:
    "Adds a node fed by node's output and moves everything that read that output onto the new node."
    output = _port(node, "outputs")
    consumers = list(_consumers(pipeline, node, output))
    added = add_node(pipeline, name, node_id, input=node, **values)
    added_output = _port(added, "outputs")
    for prim, port in consumers:
        prim.connect(port, added, added_output)
    return added


def remove_node(pipeline: Prim, node: Prim) -> None:
    "Removes node, and whatever read its output reads what fed its input instead."
    consumers = list(_consumers(pipeline, node, _port(node, "outputs")))
    if consumers:
        port = _port(node, "inputs")
        feeds = [(source, output) for each, source, output in node.connections if each == port]
        if not feeds:
            raise ValueError(f"{node.path}'s {port} isn't connected, so its readers have nothing to read")
        source = pipeline.scene.find(feeds[0][0])
        for prim, consumer in consumers:
            prim.connect(consumer, source, feeds[0][1])
    pipeline.scene.remove(node)


def bind_pipeline(prim: Prim, pipeline: Prim) -> None:
    "Makes a sensor's or a render product's captures run through pipeline."
    prim["pipelineBinding"] = [pipeline]


Scene.define_pipeline = define_pipeline  # type: ignore[attr-defined]
Prim.add_node = add_node  # type: ignore[attr-defined]
Prim.connect_output = connect_output  # type: ignore[attr-defined]
Prim.insert_after = insert_after  # type: ignore[attr-defined]
Prim.remove_node = remove_node  # type: ignore[attr-defined]
Prim.bind_pipeline = bind_pipeline  # type: ignore[attr-defined]
