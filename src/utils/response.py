from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    OutputVisualization,
)


def build_response_sift_comparison_test(context):

    outputDetections = OutputDetections(
        value=context.output_detections
    )

    outputVisualization = OutputVisualization(
        value=context.output_visualization
    )

    outputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections,
        OutputVisualization=outputVisualization
    )

    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    executor = SiftComparisonTest(
        value=response
    )

    configExecutor = ConfigExecutor(
        value=executor
    )

    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    packageModel = package.build_model(context)

    return packageModel