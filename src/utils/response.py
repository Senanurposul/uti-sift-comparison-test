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

    # Executor tarafından oluşturulan
    # Detection listesini OutputDetections
    # modeline koyuyoruz.
    output_detections = OutputDetections(
        value=context.output_detections
    )

    # Outputs modelini oluşturuyoruz.
    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections
    )

    # Response modelini oluşturuyoruz.
    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # Executor response'unu oluşturuyoruz.
    executor = SiftComparisonTest(
        value=response
    )

    # ConfigExecutor oluşturuluyor.
    configExecutor = ConfigExecutor(
        value=executor
    )

    # Package config oluşturuluyor.
    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # PackageHelper ile Novavision package response'u
    # oluşturuluyor.
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    return package.build_model(context)