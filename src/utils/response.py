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

    # Detection output
    outputDetections = OutputDetections(
        value=context.output_detections
    )

    # Visualization output
    # Visualization henüz oluşturulmamışsa None döndürür.
    outputVisualization = OutputVisualization(
        value=getattr(context, "output_visualization", None)
    )

    # Outputs
    outputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections,
        OutputVisualization=outputVisualization
    )

    # Response
    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # Executor
    executor = SiftComparisonTest(
        value=response
    )

    # Config Executor
    configExecutor = ConfigExecutor(
        value=executor
    )

    # Package Config
    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # Package
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    return package.build_model(context)