from sdks.novavision.src.helper.package import PackageHelper

from components.SIFTComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SIFTComparison,
    SIFTComparisonResponse,
    SIFTComparisonOutputs,
    OutputDetections,
    OutputVisualization,
)


def build_response_sift_comparison(context):

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

    outputs = SIFTComparisonOutputs(
        OutputDetections=output_detections,
        OutputVisualization=output_visualization
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    response = SIFTComparisonResponse(
        outputs=outputs
    )

    # ========================================================
    # EXECUTOR
    # ========================================================

    executor = SIFTComparison(
        value=response
    )

    # ========================================================
    # PACKAGE CONFIG
    # ========================================================

    configExecutor = ConfigExecutor(
        value=executor
    )

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

    return package.build_model(
        context
    )