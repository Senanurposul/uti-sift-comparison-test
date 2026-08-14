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

    # ========================================================
    # DETECTION OUTPUT
    # ========================================================

    outputDetections = OutputDetections(
        value=context.output_detections
    )

    # ========================================================
    # VISUALIZATION OUTPUT
    # ========================================================

    outputVisualization = OutputVisualization(
        value=context.visualization_image
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    siftComparisonOutputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections,
        OutputVisualization=outputVisualization
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    siftComparisonResponse = SiftComparisonTestResponse(
        outputs=siftComparisonOutputs
    )

    # ========================================================
    # EXECUTOR
    # ========================================================

    siftComparison = SiftComparisonTest(
        value=siftComparisonResponse
    )

    # ========================================================
    # CONFIG EXECUTOR
    # ========================================================

    configExecutor = ConfigExecutor(
        value=siftComparison
    )

    # ========================================================
    # PACKAGE CONFIG
    # ========================================================

    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # ========================================================
    # PACKAGE
    # ========================================================

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    packageModel = package.build_model(context)

    return packageModel