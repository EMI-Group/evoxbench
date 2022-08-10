from nasbenchbase.models import NASBenchResult
from django.db import models


class MobileNetV3Result(NASBenchResult):
    params = models.IntegerField()
    flops = models.IntegerField()
    latency = models.FloatField()
    valid_acc = models.FloatField()
    test_acc = models.FloatField()
