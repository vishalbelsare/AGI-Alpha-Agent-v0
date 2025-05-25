# Simplified AWS Fargate deployment
variable "region" { default = "us-east-1" }
variable "openai_api_key" { default = "" }
variable "agi_insight_offline" { default = "0" }
variable "agi_insight_bus_port" { default = 6006 }
variable "agi_insight_ledger_path" { default = "./ledger/audit.db" }

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
      image     = "alpha-demo:latest"
      essential = true
      environment = [
        { name = "OPENAI_API_KEY", value = var.openai_api_key },
        { name = "RUN_MODE", value = "api" },
        { name = "AGI_INSIGHT_OFFLINE", value = var.agi_insight_offline },
        { name = "AGI_INSIGHT_BUS_PORT", value = tostring(var.agi_insight_bus_port) },
        { name = "AGI_INSIGHT_LEDGER_PATH", value = var.agi_insight_ledger_path }
      ]
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
    subnets         = ["subnet-12345"]
    assign_public_ip = true
  }
}
