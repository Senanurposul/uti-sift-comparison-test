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

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_detections = OutputDetections(
        value=context.output_detections
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # --------------------------------------------------------
    # Executor
    # --------------------------------------------------------

    executor = SiftComparisonTest(
        value=response
    )

    # --------------------------------------------------------
    # Package Config
    # --------------------------------------------------------

    configExecutor = ConfigExecutor(
        value=executor
    )

    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # --------------------------------------------------------
    # Package
    # --------------------------------------------------------

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    return package.build_model(context)