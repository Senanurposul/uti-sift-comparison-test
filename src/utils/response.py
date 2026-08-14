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

    # ============================================================
    # DETECTION OUTPUT
    # ============================================================

    outputDetections = OutputDetections(
        value=context.output_detections
    )

    # ============================================================
    # VISUALIZATION OUTPUT
    # ============================================================

    outputVisualization = OutputVisualization(
        value=context.output_visualization
    )

    # ============================================================
    # OUTPUTS
    # ============================================================

    siftComparisonTestOutputs = SiftComparisonTestOutputs(
        OutputDetections=outputDetections,
        OutputVisualization=outputVisualization
    )

    # ============================================================
    # RESPONSE
    # ============================================================

    siftComparisonTestResponse = SiftComparisonTestResponse(
        outputs=siftComparisonTestOutputs
    )

    # ============================================================
    # EXECUTOR
    # ============================================================

    siftComparisonTest = SiftComparisonTest(
        value=siftComparisonTestResponse
    )

    # ============================================================
    # CONFIG EXECUTOR
    # ============================================================

    configExecutor = ConfigExecutor(
        value=siftComparisonTest
    )

    # ============================================================
    # PACKAGE CONFIG
    # ============================================================

    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # ============================================================
    # PACKAGE
    # ============================================================

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    packageModel = package.build_model(context)

    return packageModel