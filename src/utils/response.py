from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
)


def build_response_sift_comparison(context):

    outputDetections = OutputDetections(
        value=context.output_detections
    )

    siftComparisonOutputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections
    )

    siftComparisonResponse = SiftComparisonTestResponse(
        outputs=siftComparisonOutputs
    )

    siftComparison = SiftComparisonTest(
        value=siftComparisonResponse
    )

    configExecutor = ConfigExecutor(
        value=siftComparison
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