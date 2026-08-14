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

    # Detection çıktısını oluştur
    output_detections = OutputDetections(
        value=context.output_detections
    )

    # Visualization görüntüsünü oluştur
    output_visualization = OutputVisualization(
        value=context.visualization_image
    )

    # Outputs modelini oluştur
    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        OutputVisualization=output_visualization
    )

    # Response modelini oluştur
    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # Executor response'unu oluştur
    executor = SiftComparisonTest(
        value=response
    )

    # ConfigExecutor oluştur
    configExecutor = ConfigExecutor(
        value=executor
    )

    # Package config oluştur
    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # Novavision package response'u oluştur
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    return package.build_model(context)