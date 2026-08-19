from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
)


def build_response_sift_comparison_test(context):

    outputDetections = OutputDetections(
        value=context.output_detections
    )

    outputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections
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

    return package.build_model(context)