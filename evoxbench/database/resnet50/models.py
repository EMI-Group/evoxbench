from nasbenchbase.models import NASBenchResult
from django.db import models


class ResNet50Result(NASBenchResult):
    params = models.IntegerField()
    flops = models.IntegerField()
    # latency = models.FloatField()
    valid_err = models.FloatField()
    test_err = models.FloatField()
