from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    OutputVisualization1,
    OutputVisualization2,
    OutputVisualizationMatches,
)


def build_response_sift_comparison_test(context):
    output_detections = OutputDetections(value=context.output_detections)
    output_visualization_1 = OutputVisualization1(value=getattr(context,"output_visualization_1", None, ))
    output_visualization_2 = OutputVisualization2(value=getattr(context,"output_visualization_2",None,))
    output_visualization_matches = OutputVisualizationMatches(value=getattr(context,"output_visualization_matches",None,))
    outputs = SiftComparisonTestOutputs(OutputDetections=output_detections,OutputVisualization1=output_visualization_1,OutputVisualization2=output_visualization_2,OutputVisualizationMatches=output_visualization_matches,)
    response = SiftComparisonTestResponse(outputs=outputs)
    executor = SiftComparisonTest(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)
    package = PackageHelper(packageModel=PackageModel,packageConfigs=package_configs,)
    return package.build_model(context)