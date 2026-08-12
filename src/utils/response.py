from sdks.novavision.src.helper.package import PackageHelper
from components.SIFTComparison.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SIFTComparison,
    SIFTComparisonResponse,
    SIFTComparisonOutputs,
    OutputDetections,
)


def build_response_sift_comparison(context):
    output_detections = OutputDetections(value=context.output_detections)
    outputs = SIFTComparisonOutputs(
        OutputDetections=output_detections,
    )
    response = SIFTComparisonResponse(outputs=outputs)
    executor = SIFTComparison(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)