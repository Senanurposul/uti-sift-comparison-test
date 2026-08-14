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

    # ========================================================
    # Detection output
    # ========================================================

    output_detections = OutputDetections(
        value=context.output_detections
    )

    # ========================================================
    # Visualization output
    # ========================================================

    output_visualization = OutputVisualization(
        value=context.output_visualization
    )

    # ========================================================
    # Outputs
    # ========================================================

    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        OutputVisualization=output_visualization
    )

    # ========================================================
    # Response
    # ========================================================

    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # ========================================================
    # Executor
    # ========================================================

    executor = SiftComparisonTest(
        value=response
    )

    # ========================================================
    # Config Executor
    # ========================================================

    configExecutor = ConfigExecutor(
        value=executor
    )

    # ========================================================
    # Package Config
    # ========================================================

    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # ========================================================
    # Package Response
    # ========================================================

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    return package.build_model(context)