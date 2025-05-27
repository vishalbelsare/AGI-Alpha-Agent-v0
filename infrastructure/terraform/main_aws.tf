# Simplified AWS Fargate deployment
variable "region" { default = "us-east-1" }
variable "openai_api_key" { default = "" }
variable "openai_api_key_secret_id" { default = "" }
variable "agi_insight_offline" { default = "0" }
variable "agi_insight_bus_port" { default = 6006 }
variable "agi_insight_ledger_path" { default = "./ledger/audit.db" }
variable "container_image" { default = "alpha-demo:latest" }
variable "subnets" {
  type    = list(string)
  default = ["subnet-12345"]
}

provider "aws" { region = var.region }


resource "aws_ecs_cluster" "af" {
  name = "alpha-factory"
}

# Task definition with environment variables
resource "aws_ecs_task_definition" "af" {
  family                   = "alpha-factory"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  container_definitions = jsonencode([
    {
      name      = "orchestrator"
      image     = var.container_image
      essential = true
      environment = concat([
        { name = "RUN_MODE", value = "api" },
        { name = "AGI_INSIGHT_OFFLINE", value = var.agi_insight_offline },
        { name = "AGI_INSIGHT_BUS_PORT", value = tostring(var.agi_insight_bus_port) },
        { name = "AGI_INSIGHT_LEDGER_PATH", value = var.agi_insight_ledger_path }
      ],
      var.openai_api_key_secret_id == "" ? [
        { name = "OPENAI_API_KEY", value = var.openai_api_key }
      ] : [
        { name = "AGI_INSIGHT_SECRET_BACKEND", value = "aws" },
        { name = "AWS_REGION", value = var.region },
        { name = "OPENAI_API_KEY_SECRET_ID", value = var.openai_api_key_secret_id }
      ])
      portMappings = [
        { containerPort = 8000 },
        { containerPort = 6006 }
      ]
    }
  ])
}

resource "aws_ecs_service" "af" {
  name            = "alpha-factory"
  cluster         = aws_ecs_cluster.af.id
  task_definition = aws_ecs_task_definition.af.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.subnets
    assign_public_ip = true
  }
}
