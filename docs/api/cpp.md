# C++ API

The public headers are under `ernie/scene`, `ernie/session`, `ernie/app`, `ernie/geometry`, `ernie/util` and `ernie/device`, installed with Ernie and found with `find_package(Ernie)`. The classes most programs start from:

| Area | Classes |
| --- | --- |
| Scene | [Scene](../cpp/classernie_1_1Scene.md), [Prim](../cpp/classernie_1_1Prim.md), [Path](../cpp/classernie_1_1Path.md), [Value](../cpp/classernie_1_1Value.md), [Tensor](../cpp/classernie_1_1Tensor.md), [EditHistory](../cpp/classernie_1_1EditHistory.md) |
| Typed prims | [SchemaView](../cpp/classernie_1_1SchemaView.md), [PrimTypeDef](../cpp/structernie_1_1PrimTypeDef.md), [AttributeDef](../cpp/classernie_1_1AttributeDef.md), [SphereLight](../cpp/classernie_1_1SphereLight.md), [Camera](../cpp/classernie_1_1Camera.md), [RenderSettings](../cpp/classernie_1_1RenderSettings.md) |
| Rendering | [Renderer](../cpp/classernie_1_1Renderer.md), [RendererOptions](../cpp/structernie_1_1RendererOptions.md), [RenderProductOutput](../cpp/structernie_1_1RenderProductOutput.md), [ProductFrame](../cpp/structernie_1_1ProductFrame.md) |
| Cameras and lights | [CameraParams](../cpp/structernie_1_1CameraParams.md), [LensModel](../cpp/structernie_1_1LensModel.md), [CameraView](../cpp/structernie_1_1CameraView.md), [LightParams](../cpp/structernie_1_1LightParams.md), [FrameParams](../cpp/structernie_1_1FrameParams.md) |
| Extending | [BrdfRegistry](../cpp/classernie_1_1BrdfRegistry.md), [SensorFilterRegistry](../cpp/classernie_1_1SensorFilterRegistry.md), [Overlay](../cpp/classernie_1_1Overlay.md), [RenderDelegate](../cpp/classernie_1_1RenderDelegate.md) |
| Viewer | [Viewer](../cpp/classernie_1_1app_1_1Viewer.md), [Panel](../cpp/classernie_1_1app_1_1Panel.md), [WidgetPanel](../cpp/classernie_1_1app_1_1WidgetPanel.md) |
| Devices | [VulkanContext](../cpp/classernie_1_1VulkanContext.md), [ComputeInterop](../cpp/classernie_1_1ComputeInterop.md) |

Every class is under [Classes](../cpp/annotated.md), by header under [Files](../cpp/files.md), and the free functions under [Functions](../cpp/namespace_member_functions.md).
