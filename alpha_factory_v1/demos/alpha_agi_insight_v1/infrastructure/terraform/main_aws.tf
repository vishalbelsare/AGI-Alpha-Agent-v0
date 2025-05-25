# Example AWS ECS deployment using Fargate

variable "region" { default = "us-east-1" }
variable "image_tag" { default = "latest" }
variable "repository_name" { default = "alpha-insight" }
variable "vpc_id" { default = "vpc-12345" }
variable "subnets" {
  type    = list(string)
  default = ["subnet-12345"]
}

provider "aws" {
  region = var.region
}

resource "aws_ecr_repository" "insight" {
  name = var.repository_name
}

resource "aws_iam_role" "ecs_exec" {
  name               = "ecsTaskExecutionRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_exec" {
  role       = aws_iam_role.ecs_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_cluster" "insight" {
  name = "alpha-insight"
}

resource "aws_ecs_task_definition" "insight" {
  family                   = "alpha-insight"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_exec.arn
  container_definitions = jsonencode([
    {
      name      = "orchestrator"
      image     = "${aws_ecr_repository.insight.repository_url}:${var.image_tag}"
      essential = true
      environment = [{ name = "RUN_MODE", value = "api" }]
      portMappings = [
        { containerPort = 8000 },
        { containerPort = 6006 }
      ]
    }
  ])
}

resource "aws_security_group" "insight" {
  name   = "alpha-insight"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_service" "insight" {
  name            = "alpha-insight"
  cluster         = aws_ecs_cluster.insight.id
  task_definition = aws_ecs_task_definition.insight.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.subnets
    security_groups = [aws_security_group.insight.id]
    assign_public_ip = true
  }
}
