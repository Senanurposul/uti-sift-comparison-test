from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    Visualization1,
    Visualization2,
    VisualizationMatches,
)


def build_response_sift_comparison_test(context):

    # --------------------------------------------------------
    # Output Detections
    # --------------------------------------------------------

    output_detections = OutputDetections(
        value=context.output_detections
    )

    # --------------------------------------------------------
    # Visualization outputs
    # --------------------------------------------------------

    visualization_1 = Visualization1(
        value=context.visualization_1
    )

    visualization_2 = Visualization2(
        value=context.visualization_2
    )

    visualization_matches = VisualizationMatches(
        value=context.visualization_matches
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        Visualization1=visualization_1,
        Visualization2=visualization_2,
        VisualizationMatches=visualization_matches
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

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