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

    Login to the AWS Console and navigate to [SageMaker](https://console.aws.amazon.com/sagemaker/). You can find SageMaker in the Machine Learning section or using the search box at the top of the console. The SageMaker Dashboard contains links to all major components: Notebook, Training, and Inference. Second on the Notebook list is "Lifecycle configurations". Click on that link.

    Lifecycle configurations are startup scripts that initialize your Jupyter notebook environments. They can be run once on creation or on every notebook startup.

    Click on the orange button labelled "Create Configuration".

    Under "Scripts" click on "Create notebook".

    Name the lifecycle configuration ```rec-engine-workshop-lc```. In the *Scripts* section click on **Create notebook**. Place your cursor on the line under the initial bash commands and paste the following:

    ```
    cd SageMaker
    git clone https://github.com/aws-samples/recommendation-engine-workshop.git

    mkdir Recommendation-Workshop
    mv recommendation-engine-workshop/Lab\ 1\ -\ Introduction\ to\ Factorization\ Machine/ Recommendation-Workshop
    mv recommendation-engine-workshop/Lab\ 2\ -\ Introduction\ to\ Object2Vec/ Recommendation-Workshop
    mkdir Recommendation-Workshop/Lab\ 1\ -\ Introduction\ to\ Factorization\ Machine/LambdaPackage
    mkdir Recommendation-Workshop/Lab\ 2\ -\ Introduction\ to\ Object2Vec/LambdaPackage
    mv Recommendation-Workshop/Lab\ 1\ -\ Introduction\ to\ Factorization\ Machine/lambda_function.py Recommendation-Workshop/Lab\ 1\ -\ Introduction\ to\ Factorization\ Machine/LambdaPackage/
    mv Recommendation-Workshop/Lab\ 2\ -\ Introduction\ to\ Object2Vec/lambda_function.py Recommendation-Workshop/Lab\ 2\ -\ Introduction\ to\ Object2Vec/LambdaPackage/

    chown -R ec2-user.ec2-user Recommendation-Workshop/
    rm -rf recommendation-engine-workshop

    wget https://www.mirrorservice.org/sites/dl.fedoraproject.org/pub/epel/7/x86_64/Packages/p/p7zip-16.02-10.el7.x86_64.rpm
    wget https://www.mirrorservice.org/sites/dl.fedoraproject.org/pub/epel/7/x86_64/Packages/p/p7zip-plugins-16.02-10.el7.x86_64.rpm

    rpm -U --quiet p7zip-16.02-10.el7.x86_64.rpm
    rpm -U --quiet p7zip-plugins-16.02-10.el7.x86_64.rpm

    rm p7zip-16.02-10.el7.x86_64.rpm
    rm p7zip-plugins-16.02-10.el7.x86_64.rpm

    ```

    The above commands do the following when the instance is created:
    - Download the code and necessary files from the workshop GitHub repo.
    - Organize the folder structure and place files in session folders.
    - Set write permission to the folders
    - Install 7-Zip which is required to compress lambda packages to their smallest size

1. Click on "Notebook instances". Click on "Create notebook instance" to create a Jupyter notebook using the lifecycle configuration created inthe previous step.

    ![Notebook Instance](images/notebook-instance.jpg)

    - Take note of the region in which you are running SageMaker. You'll need to recall this region when you create an S3 bucket in the next step. For this workshop we're using Oregon: ```us-west-2```.
    - Name the instance as ```rec-engine-workshop```.
    - Choose instance type such as ```ml.m5.4xlarge```.
    - Under IAM role choose "Create a new role"
            - Choose "Any S3 bucket"
            - Click "Create role" and take note of the newly created role.
    - No VPC
    - Choose lifecycle configuration, ```rec-engine-workshop-lc```
    - No Custom Encryption
    - Click on **Create notebook instance**.

    It takes about 3 minutes for a SageMaker notebook instance to provision. During this time you'll see the status *Pending*.

1. Navigate to [S3](https://console.aws.amazon.com/s3) on the AWS Console. While we're waiting for the notebook to be provisioned, let's create an S3 bucket with a globally-unique name, such as: ```rec-engine-workshop-yourname```. Take care to choose the same region for your bucket as your SageMaker notebook. For this workshop we're using Oregon: ```us-west-2```.

    This bucket is necessary to store the training data and models you're creating in this workshop. Take note of the region. SageMaker must be run in the same region as your newly created S3 bucket. If for any reason you choose an alternate region simply ensure that SageMaker runs in the same region as your newly created bucket.

1. Setup IAM roles and attach policies

    We need to add rights to our newly-created SageMaker role and create a new role for Lambda functions that will invoke model endpoints.  We'll be using 2 policies, ```LambdaFullAccess```, and ```S3FullAccess```. These permissions are required in the notebook as we will be uploading objects to S3 and creatingLambda functions. *Note: These are highly permissive settings which are suitable for this workshop, however, far too broad for commercial production.*

    ![SageMaker IAM](images/sagemaker-iam.jpg)

    - Navigate to IAM in the AWS Console. From the left navigation click on "Roles".
    - On the seach box, type ```SageMaker```. The name of the role looks like ```AmazonSageMaker-ExecutionRole-20181022T083494```
    - On the Summary page, click on the Attach policies button.
    - Use the search box to add ```LambdaFullAccess```, ```S3FullAccess``` individually. You do this by searching, clicking on the on the checkbox, searching again, and selecting the next policy. Your checkbox selections persist through multiple searches.
    - After you have added both policies click on Attach policy button.

1. While we are on the IAM console, let's create the Lambda execution role. This role is for our Lambda functions to have sufficient permissions to perform tasks such as getting objects from S3 and invoking an endpoint that is deployed in SageMaker.

     ![Lambda IAM](images/lambda-iam.jpg)

    - Navigate to IAM in the AWS Console. From the left navigation click on "Roles". Click on "Create Role".
    - From the list of AWS Services choose: Lambda. Click on Next:Permissions.
    - Use the search box to add ```S3FullAccess```, and ```AmazonSageMakerFullAccess``` individually. You do this by searching, clicking onthe on the checkbox, searching again, and selecting the next policy. Note that this is a highly permissive configuration that should not be used in production. It is only intended to facilitate learning in this workshop.
    - After you have added all 3 policies click on Next:Tags, and then Next:Review.
    - For Role Name use Lambda_ServerlessWorkshop_Role.
    - Click on Create role.
    - Return to the SageMaker console.

1. By now the notebook instance is ready, open the instance by clicking "Open Jupyter". Take moment to browse the folder structures created.

1. Open the [MXNet Deep Matrix Factorization Notebook](MXNet_Deep_Matrix_Factorization.ipynb) Take your time and read the theoretical descriptions and examine the code before proceeding to the next steps.

1. Now move on to [Lab 2: Introduction to Object2Vec](../Lab2\ -\ Introduction\ to\ Object2Vec). 

## Clean Up!
**If you are not moving on the Lab 2** please clean up to avoid unnecessary charges. Cleanup instructions are found at the end of [Lab 2: Introduction to Object2Vec](../Lab2%20-%20Introduction%20to%20Object2Vec). 

