# Deploying Guardrails Server on AWS

Guardrails allows you to deploy a dedicated server to run guard executions while continuing to use the Guardrails SDK as you do today.

In this cookbook we show an example of deploying a containerized version of Guardrails API into AWS leveraging AWS ECS. 

> Note: Guardrails API is a feature is available as of `>=0.5.0`


# Step 1: Containerizing Guardrails API

Create a `Dockerfile` in a new working directory `guardrails-server` (or alternative)

```Dockerfile
FROM python:3.11-slim

ARG GUARDRAILS_TOKEN
ARG GUARDRAILS_SDK_VERSION="guardrails-ai[api]>=0.5.0,<6"

# Set environment variables to avoid writing .pyc files and to unbuffer Python output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOGLEVEL="DEBUG"
ENV GUARDRAILS_LOG_LEVEL="DEBUG"
ENV APP_ENVIRONMENT="production"

WORKDIR /app

# Install Git and necessary dependencies
RUN apt-get update && \
    apt-get install -y git curl gcc jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install guardrails, the guardrails API, and gunicorn
RUN pip install $GUARDRAILS_SDK_VERSION "gunicorn"

RUN guardrails configure --enable-metrics --enable-remote-inferencing --token $GUARDRAILS_TOKEN

# Copy the hub requirements file to the container
COPY hub-requirements.txt /app/hub-requirements.txt

# Install hub dependencies
RUN export GR_INSTALLS=$(python -c 'print(",".join(open("./hub-requirements.txt", encoding="utf-8").read().splitlines()))'); \
    set -e; IFS=','; \
    for url in $GR_INSTALLS; do \
        if [ -n "$url" ]; then \
            guardrails hub install --quiet --no-install-local-models "$url"; \
        fi; \
    done

# Overwrite Config file with one with user's settings
COPY config.py /app/config.py

# Copy the guardrails API config file to the installed package
RUN cp /app/config.py $(pip show guardrails-api | grep Location | awk '{print $2}')/guardrails_api/config.py

# Expose port 8000 for the application
EXPOSE 8000

# Command to start the Gunicorn server with specified settings
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout=5", "--threads=8", "guardrails_api.app:create_app()"]
```

Then create a `hub-requirements.txt` file containing all of the hub installs required for your Guards:

*Example:*
```
hub://guardrails/competitor_check
hub://guardrails/toxic_language
```

At this point you can now generate the required config file (`config.py`) which is needed for the server to know how to configure each guard.

```bash
guardrails create --validators=$(python -c 'print(",".join(open("./hub-requirements.txt", encoding="utf-8").read().splitlines()))')
```

> ⚠️ *Important:* Ensure to add in any telemetry configuration and guard configuration to the config file AND ensure each guard is given a variable name i.e `guard1 = Guard(...)`

# Step 2: Deploying Infra

By leveraging AWS ECS we can scale to handle increasing workloads by scaling the number of containers. Furthermore we can leverage a streamlined deployment process using ECS with rolling updates (Step 3).

We can now deploy the infrastructure needed for AWS ECS which includes:
- Networking Resources (VPC, Load Balancer, Security Groups, Subnets, ect)
- IAM Roles & Policies (ECS Task & Execution Role)
- ECS Cluster (ECS Service, Task, Task Definition)

We start by initilizing terraform with:

```bash
terraform init
```

One can then copy the provided [Terraform Code](#Terraform) or use their own by placing into our working directory and running:

```bash
terraform apply -var="aws_region=us-east-1" -var="backend_memory=16384" -var="backend_cpu=8192" -var="desired_count=0"
```

> Each can be configured based on your requirements. `desired_count` corresponds to the number of containers that should always be running. Alternatively one can configure a minimum & maximum count with some autoscaling policy. It is initially set to `0` since we have yet to upload the container to the AWS container registry (ECR).

Once the deployment has succeeded you should see some output values (which will be required if you wish to set up CI).

# Step 3

You can deploy the application manually as specified in Step 3a or skip to Step 3b to deploy the application with CI.

## Step 3a: Deploying Application Manually

Firstly, create or use your existing guardrails token and export it to your current shell `export GUARDRAILS_TOKEN="..."`

```bash
# Optionally use the command below to use your existing token
export GUARDRAILS_TOKEN=$(cat ~/.guardrailsrc| awk -F 'token=' '{print $2}' | awk '{print $1}' | tr -d '\n')
```

Run the following to build your container and push up to ECR:

```bash
docker build --platform linux/amd64 --build-arg GUARDRAILS_TOKEN=$GUARDRAILS_TOKEN -t guardrails-api:latest .

aws ecr get-login-password --region ${YOUR_AWS_REGION} | docker login --username AWS --password-stdin ${YOUR_AWS_ACCOUNT_ID}.dkr.ecr.${YOUR_AWS_REGION}.amazonaws.com

docker tag guardrails-api:latest ${YOUR_AWS_ACCOUNT_ID}.dkr.ecr.${YOUR_AWS_REGION}.amazonaws.com/gr-backend-images:latest

docker push ${YOUR_AWS_ACCOUNT_ID}.dkr.ecr.${YOUR_AWS_REGION}.amazonaws.com/gr-backend-images:latest
```

Lastly you can deploy your ECS service by scaling up the container count using the AWS CLI:

```bash
aws ecs update-service --cluster gr-backend-ecs-cluster --service gr-backend-ecs-service --desired-count 3 --force-new-deployment
```


## Step 3b: Deploying Application Using CI

In some cases you may update your `config.py` file or your `hub-requirements.txt` file to add/remove guards. Here it may help to set up CI to automate the process of updating your Guardrails API container and deploy to ECS.

For this to work you must set the following configuration on Github Actions:
- `AWS_ACCESS_KEY_ID` [Secret]
- `AWS_SECRET_ACCESS_KEY` [Secret]
- `AWS_ECR_REPOSITORY` [Variable]
- `AWS_ECS_CLUSTER_NAME` [Variable] Can be set to `gr-backend-ecs-cluster` as default
- `AWS_ECS_SERVICE_NAME` [Variable] Can be set to `gr-backend-ecs-service` as default

Here is some provided Github Actions workflow which does this on pushes to the `main` branch:


```yaml
name: Deploy Backend

on:
  push:
    branches:
      - main

jobs:
  deploy_app:
    name: Deploy Backend
    runs-on: ubuntu-latest
    env:
      AWS_REGION: us-east-1
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      AWS_ECR_REPOSITORY: ${{ vars.AWS_ECR_REPOSITORY }} 
      AWS_ECS_CLUSTER_NAME: ${{ vars.AWS_ECS_CLUSTER_NAME }}
      AWS_ECS_SERVICE_NAME: ${{ vars.AWS_ECS_SERVICE_NAME }}
      WORKING_DIR: "./"
    steps: 
    
      - name: Check out head
        uses: actions/checkout@v3
        with:
          persist-credentials: false

      - name: Set up QEMU
        uses: docker/setup-qemu-action@master
        with:
          platforms: linux/amd64

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@master
        with:
          platforms: linux/amd64

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-region: ${{ env.AWS_REGION }}
          aws-secret-access-key: ${{ env.AWS_SECRET_ACCESS_KEY }}
          aws-access-key-id: ${{ env.AWS_ACCESS_KEY_ID }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build & Push ECR Image
        uses: docker/build-push-action@v2
        with:
          builder: ${{ steps.buildx.outputs.name }}
          context: ${{ env.WORKING_DIR }}
          platforms: linux/amd64
          cache-from: type=gha
          cache-to: type=gha,mode=max
          push: true
          tags: ${{ env.AWS_ECR_REPOSITORY }}:latest

      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster ${{ env.AWS_ECS_CLUSTER_NAME }} --service ${{ env.AWS_ECS_SERVICE_NAME }} --desired-count 1 --force-new-deployment
        env:
          AWS_DEFAULT_REGION: ${{ env.AWS_REGION }}


```

# Step 4: Using with SDK

You should be able to get the URL for your Guardrails API using:

```bash
export GUARDRAILS_BASE_URL=$(terraform output -raw backend_service_url)
echo "http://$GUARDRAILS_BASE_URL"
```

By setting the above environment variable `GUARDRAILS_BASE_URL` the SDK will be able to use this as a backend for running validations.


# Considerations for Production

When building the container we made some crucial choices in order to make Guardrails production ready. Namely we opted for remote inferencing & did not include local models. This allows us to reduce the image size by preventing heavy validators from being included in the final container image and instead use hosted versions of ML based validators.

In addition to the configuration above we recommend that you host your ECS cluster in private subnet and configure autoscaling policies with appropriate minimum & maximum container counts based on your expected workloads. Furthermore, we recommend that you also serve the API using https.

# Terraform

```hcl
locals {
    deployment_name = "gr-backend"   
}

variable "aws_region" {
  description = "AWS region to deploy the resources"
  type        = string
  default     = "us-east-2"
}

variable "backend_cpu" {
  description = "CPU units for the service"
  type        = number
  default     = 8*1024
}

variable "backend_memory" {
  description = "Memory units for the service"
  type        = number
  default     = 16*1024
}

variable "backend_server_port" {
    description = "Port on which the backend server listens"
    type        = number
    default     = 8000
}

variable "desired_count" {
    description = "Number of tasks to run"
    type        = number
    default     = 0
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Offering    = "Guardrails Backend"
      Vendor      = "Guardrails"
      Terraform   = "True"
    }
  }
}

################# Networking Resources

data "aws_availability_zones" "available" {}


resource "aws_vpc" "backend" {
  cidr_block            = "10.0.0.0/16"
  enable_dns_hostnames  = true

  tags = {
    Name = "${local.deployment_name}-vpc"
  }
}

resource "aws_subnet" "backend_public_subnets" {
  count = 3
  vpc_id = aws_vpc.backend.id
  cidr_block              = cidrsubnet("10.0.0.0/16", 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${local.deployment_name}-public-subnet-${count.index}"
  }
}

resource "aws_eip" "backend" {
  count      = 2
  vpc        = true
  depends_on = [aws_internet_gateway.backend]
}

resource "aws_nat_gateway" "backend" {
  count         = 2
  subnet_id     = aws_subnet.backend_public_subnets[count.index].id
  allocation_id = aws_eip.backend[count.index].id
}

resource "aws_internet_gateway" "backend" {
  vpc_id = aws_vpc.backend.id

  tags = {
    Name = "${local.deployment_name}-igw"
  }
}

resource "aws_route_table" "backend_public_routes" {
  vpc_id = aws_vpc.backend.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.backend.id
  }
}

resource "aws_route_table_association" "backend_public_routes" {
  count          = length(aws_subnet.backend_public_subnets)
  subnet_id      = aws_subnet.backend_public_subnets[count.index].id
  route_table_id = aws_route_table.backend_public_routes.id
}

resource "aws_lb" "app_lb" {
  name                             = "${local.deployment_name}-nlb"
  load_balancer_type               = "network"
  internal                         = false
  subnets                          = aws_subnet.backend_public_subnets[*].id
  enable_cross_zone_load_balancing = false
}

resource "aws_lb_listener" "app_lb_listener" {
  load_balancer_arn = aws_lb.app_lb.arn

  protocol = "TCP"
  port     = 80

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_lb.arn
  }
}

resource "aws_lb_target_group" "app_lb" {
  name        = "${local.deployment_name}-nlb-tg"
  protocol    = "TCP"
  port        = 80
  vpc_id      = aws_vpc.backend.id
  target_type = "ip"

  health_check {
    healthy_threshold   = "2"
    interval            = "30"
    protocol            = "HTTP"
    timeout             = "3"
    unhealthy_threshold = "3"
    path                = "/"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "backend" {
  name        = "${local.deployment_name}-firewall"
  description = "Guardrails backend firewall"
  vpc_id      = aws_vpc.backend.id

  ingress {
    description = "Guardrails API Access"
    from_port   = var.backend_server_port
    to_port     = var.backend_server_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  revoke_rules_on_delete = true
}

################# Log Resources

resource "aws_cloudwatch_log_group" "backend_log_group" {
  name              = "${local.deployment_name}-log-group"
  retention_in_days = 30
}

################# Application Resources

resource "aws_ecr_repository" "backend_images" {
  name = "${local.deployment_name}-images"
}

resource "aws_ecs_cluster" "backend" {
  name = "${local.deployment_name}-ecs-cluster"

  configuration {
    execute_command_configuration {
      logging    = "OVERRIDE"

      log_configuration {
        cloud_watch_encryption_enabled = false
        cloud_watch_log_group_name     = aws_cloudwatch_log_group.backend_log_group.name
      }
    }
  }

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

data "aws_caller_identity" "current" {}

resource "aws_ecs_task_definition" "backend" {
  family                = "${local.deployment_name}-backend-task-defn"
  execution_role_arn    = aws_iam_role.ecs_execution_role.arn
  task_role_arn         = aws_iam_role.ecs_task_role.arn
  network_mode          = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                   = var.backend_cpu
  memory                = var.backend_memory

  container_definitions = jsonencode([
    {
      name             = "${local.deployment_name}-task",
      image            = "${aws_ecr_repository.backend_images.repository_url}:latest",
      cpu              = var.backend_cpu,
      memory           = var.backend_memory,
      networkMode      = "awsvpc",

      portMappings     = [
        {
          containerPort = var.backend_server_port,
          hostPort      = var.backend_server_port,
          protocol      = "tcp"
        }
      ],
      logConfiguration = {
        logDriver = "awslogs",
        options   = {
          "awslogs-group"         = aws_cloudwatch_log_group.backend_log_group.name,
          "awslogs-region"        = var.aws_region,
          "awslogs-stream-prefix" = "backend"
        }
      },
      linuxParameters  = {
        initProcessEnabled = true
      },
      healthCheck      = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.backend_server_port}/ || exit 1"],
        interval    = 30,
        startPeriod = 30,
        timeout     = 10,
        retries     = 3
      },
      environment      = [
        {
          name  = "AWS_ACCOUNT_ID",
          value = data.aws_caller_identity.current.account_id
        },
        {
          name  = "HOST",
          value = "http://${aws_lb.app_lb.dns_name}"
        },
        {
          name  = "SELF_ENDPOINT",
          value = "http://${aws_lb.app_lb.dns_name}:${var.backend_server_port}"
        }
      ],
      essential        = true
    }
  ])
}


resource "aws_ecs_service" "backend" {
  name            = "${local.deployment_name}-ecs-service"
  cluster         = aws_ecs_cluster.backend.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  enable_execute_command = true
  wait_for_steady_state  = true

  network_configuration {
    security_groups  = [aws_security_group.backend.id]
    subnets          = aws_subnet.backend_public_subnets[*].id
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app_lb.id
    container_name   = "${local.deployment_name}-task"
    container_port   = var.backend_server_port
  }

  lifecycle {
    ignore_changes = [task_definition]
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

################# IAM Roles and Policies

resource "aws_iam_role" "ecs_execution_role" {
  name = "${local.deployment_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Effect = "Allow"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${local.deployment_name}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Effect = "Allow"
      },
    ]
  })
}

output "ecr_repository_url" {
  value = aws_ecr_repository.backend_images.repository_url
}

output "backend_service_url" {
  value = aws_lb.app_lb.dns_name
}
```