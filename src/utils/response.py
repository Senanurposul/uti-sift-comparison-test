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
    # OUTPUT DETECTIONS
    # ========================================================

    output_detections = OutputDetections(
        value=context.output_detections
    )

    # ========================================================
    # OUTPUT VISUALIZATION
    # ========================================================

    output_visualization = OutputVisualization(
        value=getattr(
            context,
            "output_visualization",
            None
        )
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        OutputVisualization=output_visualization
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # ========================================================
    # EXECUTOR
    # ========================================================

    executor = SiftComparisonTest(
        value=response
    )

    # ========================================================
    # PACKAGE CONFIG
    # ========================================================

    config_executor = ConfigExecutor(
        value=executor
    )

    package_configs = PackageConfigs(
        executor=config_executor
    )

    # ========================================================
    # PACKAGE
    # ========================================================

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )

    return package.build_model(context)