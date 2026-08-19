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

    siftComparisonTestOutputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections
    )

    siftComparisonTestResponse = SiftComparisonTestResponse(
        outputs=siftComparisonTestOutputs
    )

    siftComparisonTest = SiftComparisonTest(
        value=siftComparisonTestResponse
    )

    configExecutor = ConfigExecutor(
        value=siftComparisonTest
    )

    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    packageModel = package.build_model(
        context
    )

    return packageModel