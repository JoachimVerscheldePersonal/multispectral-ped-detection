from azure.ai.ml import command, Input, MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext, AmlCompute, Model
from azure.ai.ml.constants import AssetTypes
import configparser
# read azure credentials from local config file
config = configparser.ConfigParser()
config.read("config.ini")

# Login to configure your workspace and resource group.
credential = DefaultAzureCredential()

# Get a handle to the workspace. You can find the info on the workspace tab on ml.azure.com
ml_client = MLClient(
    credential=credential,
    subscription_id= config["AZURE"]["SubscriptionId"],
    resource_group_name=config["AZURE"]["ResourceGroupName"],
    workspace_name=config["AZURE"]["WorkspaceName"],
)

env_docker_context = Environment(
    build=BuildContext(path="azure/environment"),
    name="yolo_v8_env",
    description="environment to use ultralytics yolo v8 models",
)

ml_client.environments.create_or_update(env_docker_context)

cluster = AmlCompute(
    name="ped-detection-compute",
    type="amlcompute",
    size="Standard_E4ds_v4",
    location="westeurope",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
)

ml_client.begin_create_or_update(cluster).result()

# Register model
file_model = Model(
    path="last_blended.pt",
    type=AssetTypes.CUSTOM_MODEL,
    name="last_blended",
    description="the control model after training for 50 epochs.",
)
ml_client.models.create_or_update(file_model)

command_job = command(
    inputs=dict(
        data=Input(
            type="uri_folder",
            path="azureml:multispectral_pedestrian_detection_blended_dataset:1",
        ),
        model_to_train=Input(
            type="custom_model",
            path="azureml:last_blended:1"
        )
    ),
    code=".",
    command="""
    echo "The data asset path is ${{ inputs.data }}" &&
    # Update config.yaml to contain the correct path
    sed -i "s|path:.*$|path: ${{ inputs.data }}|" config_blended.yaml &&
    # Now config.yaml contains the correct path so we can run the training
    yolo detect train data=config_blended.yaml model=last_blended.pt batch=32 epochs=10 imgsz=640 seed=42 project=multispectral-ped-detection name=multispectral-ped-detection
    """,
    environment="azureml:yolo_v8_env:3",
    compute="ped-detection-compute",
    experiment_name="multispectral-ped-detection",
    display_name="train-blended-model",
)

# Submit the command
ml_client.jobs.create_or_update(command_job)