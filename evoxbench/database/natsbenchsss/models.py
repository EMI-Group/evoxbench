from email.policy import default
from nasbenchbase.models import NASBenchResult
from django.db import models


class NATSBenchResult(NASBenchResult):
    cifar10_valid = models.JSONField()
    cifar10 = models.JSONField()
    cifar100 = models.JSONField()
    ImageNet16_120 = models.JSONField()
    index = models.IntegerField()
    cost = models.JSONField(default=dict)
