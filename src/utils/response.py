from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    OutputVisualization1,
    OutputVisualization2,
    OutputVisualizationMatches,
)


def build_response_sift_comparison_test(context):

    # ========================================================
    # OUTPUT DETECTIONS
    # ========================================================

    output_detections = OutputDetections(
        value=context.output_detections
    )

    # ========================================================
    # VISUALIZATION 1
    # ========================================================

    output_visualization_1 = OutputVisualization1(
        value=getattr(
            context,
            "output_visualization_1",
            None
        )
    )

    # ========================================================
    # VISUALIZATION 2
    # ========================================================

    output_visualization_2 = OutputVisualization2(
        value=getattr(
            context,
            "output_visualization_2",
            None
        )
    )

    # ========================================================
    # VISUALIZATION MATCHES
    # ========================================================

    output_visualization_matches = OutputVisualizationMatches(
        value=getattr(
            context,
            "output_visualization_matches",
            None
        )
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        OutputVisualization1=output_visualization_1,
        OutputVisualization2=output_visualization_2,
        OutputVisualizationMatches=output_visualization_matches
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