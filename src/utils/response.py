from sdks.novavision.src.helper.package import PackageHelper
from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonExecutor,
    SiftComparisonExecutorResponse,
    SiftComparisonExecutorOutputs,
    OutputDetections,
    OutputMatchesImage
)


def build_response_sift_comparison(context):
    outputDetections = OutputDetections(value=context.output_detections)

    outputs_dict = {
        "outputDetections": outputDetections
    }

    if hasattr(context, "output_matches_image") and context.output_matches_image is not None:
        outputs_dict["outputMatchesImage"] = OutputMatchesImage(value=context.output_matches_image)

    siftComparisonExecutorOutputs = SiftComparisonExecutorOutputs(**outputs_dict)
    siftComparisonExecutorResponse = SiftComparisonExecutorResponse(outputs=siftComparisonExecutorOutputs)
    siftComparisonExecutor = SiftComparisonExecutor(value=siftComparisonExecutorResponse)
    configExecutor = ConfigExecutor(value=siftComparisonExecutor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)

    return packageModel