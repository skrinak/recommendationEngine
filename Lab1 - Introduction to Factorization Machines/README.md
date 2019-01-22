# Introduction to Factorization Machines

In this workshop you will develop a recommendation engine using Apache MXNet and deep matrix factorization. The dataset is from the MovieLens site (https://movielens.org/), and is comprised of movie ratings and Netflix user preferences. 

## Prerequisites
### AWS Account

To complete this workshop you'll need an AWS Account with access rights to S3, SageMaker, and to create AWS IAM roles. The code and instructions in this workshop assume only one participant is using a given AWS account at a time. If you try sharing an account with another participant, you'll run into naming conflicts for certain resources. You can work around these by appending a unique suffix to the resources that fail to create due to naming conflicts. However, these instructions do not provide details on the changes required to make this work.

All of the resources you will launch as part of this workshop are eligible for the AWS free tier if your account is less than 12 months old. See the AWS Free Tier page for more details.

### Browser

We recommend you use the latest version of Firefox or Chrome to complete this workshop.

## Steps

1. Create a SageMaker lifecycle configuration

    Login to the [AWS Console](https://aws.amazon.com/console/) and navigate to [SageMaker](https://console.aws.amazon.com/sagemaker/). You can find SageMaker in the Machine Learning section or using the search box at the top of the console. The SageMaker Dashboard contains links to all major components: Notebook, Training, and Inference. 
    
    For this workshop we'll be working out of the AWS Ireland region. Take a look at the upper right hand corner of the console. That's where you'll see an indication of your current region. If you don't see Ireland, click on it now and select EU(Ireland). You need to know the region you're working in whenever you use AWS.

    The second item in the left-hand notebook navigation section is "Lifecycle configurations". Click on that link.

    Lifecycle configurations are startup scripts that initialize your Jupyter notebook environments. They can be run once on creation or on every notebook startup.

    Click on the orange button labelled "Create Configuration".

    Under "Scripts" click on "Create notebook".

    Name the lifecycle configuration ```rec-engine-workshop-lc```. In the *Scripts* section click on **Create notebook**. Place your cursor on the line under the initial bash commands and paste the following under ```set -e```:

    ```
    cd SageMaker
    git clone https://github.com/skrinak/recommendationEngine.git
 
    chown -R ec2-user.ec2-user recommendationEngine
    ```

    Click **Create Configuration**.

    The above commands do the following when the instance is created:
    - Download the code and necessary files from the workshop GitHub repo.
    - Organize the folder structure and place files in session folders.
    - Set write permission to the folders

1. Click on "Notebook instances". Click on "Create notebook instance" to create a Jupyter notebook using the lifecycle configuration created in the previous step.

    ![Notebook Instance](images/notebook-instance.jpg)

    - Name the instance as ```rec-engine-workshop```.
    - Choose instance type such as ```ml.c5.4xlarge```. *Note: if your AWS account is less than 24-hours old you won't have permission to use this class of service. Please use the default: ml.t2.medium.*
    - Under IAM role choose "Create a new role"
            - Choose "Any S3 bucket"
            - Click "Create role" and take note of the newly created role.
    - No VPC
    - Choose lifecycle configuration, ```rec-engine-workshop-lc```
    - No Custom Encryption
    - Click on **Create notebook instance**.

    It takes about 3 minutes for a SageMaker notebook instance to provision. During this time you'll see the status *Pending*.

1. Navigate to [S3](https://console.aws.amazon.com/s3) on the AWS Console. While we're waiting for the notebook to be provisioned, let's create an S3 bucket with a globally-unique name, such as: ```rec-engine-workshop-yourname```. Take care to choose the same region for your bucket as your SageMaker notebook. For this workshop we're using Ireland.

    This bucket is necessary to store the training data and models you're creating in this workshop. Take note of the region. SageMaker must be run in the same region as your newly created S3 bucket. If for any reason you choose an alternate region simply ensure that SageMaker runs in the same region as your newly created bucket.

1. By now the notebook instance is ready, open the instance by clicking "Open Jupyter". 

1. Go to [Lab 1: Introduction to Factorization Machines](MXNet_Deep_Matrix_Factorization.ipynb). Take a moment to read the instructions and examine the code before proceeding to the next steps.

 ## Congratulations!

You've successfully created a recommendation engine using Deep Matrix Factorization. Let's move now to [Lab 2: Introduction to Object2Vec](../Lab2%20-%20Introduction%20to%20Object2Vec)
