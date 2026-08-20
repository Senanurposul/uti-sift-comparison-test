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
    output_detections = OutputDetections(value=context.output_detections)

    # Görselleştirme üretilmediyse (EnableVisualization=False
    # veya görüntüler sağlanmadıysa) value None olarak kalır.
    output_visualization = OutputVisualization(
        value=getattr(context, "output_visualization", None)
    )

    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        OutputVisualization=output_visualization,
    )
    response = SiftComparisonTestResponse(outputs=outputs)
    executor = SiftComparisonTest(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)