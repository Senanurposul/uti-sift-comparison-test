from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    OutputVisualization,
)


def build_response_sift_comparison(context):

    # Detection output
    outputDetections = OutputDetections(
        value=context.output_detections
    )

    # Visualization output
    outputVisualization = OutputVisualization(
        value=context.visualization_image
    )

    # Outputs
    siftComparisonOutputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections,
        OutputVisualization=outputVisualization
    )

    # Response
    siftComparisonResponse = SiftComparisonTestResponse(
        outputs=siftComparisonOutputs
    )

    # Executor
    siftComparison = SiftComparisonTest(
        value=siftComparisonResponse
    )

    # ConfigExecutor
    configExecutor = ConfigExecutor(
        value=siftComparison
    )

    # Package config
    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # Package
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    packageModel = package.build_model(context)

    return packageModel