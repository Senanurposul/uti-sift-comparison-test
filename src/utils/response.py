from sdks.novavision.src.helper.package import PackageHelper
from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    OutputMatchesImage,
)


def build_response_sift_comparison_test(context):
    output_detections = OutputDetections(value=context.output_detections)

    outputs_dict = {
        "OutputDetections": output_detections,
    }

    # Eğer görselleştirme üretildiyse yanıta eklenir
    if hasattr(context, "output_matches_image") and context.output_matches_image is not None:
        outputs_dict["OutputMatchesImage"] = OutputMatchesImage(
            value=context.output_matches_image
        )

    outputs = SiftComparisonTestOutputs(**outputs_dict)
    response = SiftComparisonTestResponse(outputs=outputs)
    executor = SiftComparisonTest(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)